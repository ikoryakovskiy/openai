from grlgym.envs.grl import GRLEnv

import argparse
import time
import os
import yaml
import numpy as np
import logging
from my_monitor import MyMonitor
from baselines import logger
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

import training
from my_ddpg_models import MyActor, MyCritic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

import gym
import tensorflow as tf
from mpi4py import MPI


def cfg_run(**kwargs): 
    with open("{}.yaml".format(kwargs['output']), 'w', encoding='utf8') as file:
        yaml.dump(kwargs, file, default_flow_style=False, allow_unicode=True)
    del kwargs['cores']
    run(**kwargs)

def run(cfg, seed, noise_type, layer_norm, evaluation, architecture, **kwargs):    
   
    if MPI.COMM_WORLD.Get_rank() == 0:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        logger.configure(dir_path, ['stdout'])
        
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = GRLEnv(cfg)
    gym.logger.setLevel(logging.WARN)
    env = MyMonitor(env, os.path.join(logger.get_dir(), kwargs['output']))

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev, theta = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), dt=0.03,
                                                        sigma=float(stddev) * np.ones(nb_actions), 
                                                        theta=float(theta) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = MyCritic(layer_norm=layer_norm, architecture=architecture)
    actor = MyActor(nb_actions, layer_norm=layer_norm, architecture=architecture)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, param_noise=param_noise, action_noise=action_noise,
                   actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--cfg', type=str, default='cfg/rbdl_py_balancing.yaml')
    parser.add_argument('--architecture', type=str, default='Divyam')
    parser.add_argument('--tau', type=float, default=0.001)
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-trials', type=int, default=None)
    parser.add_argument('--nb-train-steps', type=int, default=1)  # per epoch cycle and MPI worker
    parser.add_argument('--test-interval', type=int, default=10)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=1000)
    boolean_flag(parser, 'evaluation', default=True)
    parser.add_argument('--output', type=str, default='default')
    parser.add_argument('--load-file', type=str, default='')
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    
    # Run actual script.
    cfg_run(**args)
