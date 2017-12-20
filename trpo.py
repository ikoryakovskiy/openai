#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from grlgym.envs.grl import GRLEnv

from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import logging
import gym
import os
import yaml
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import baselines.common.tf_util as U
from my_monitor import MyMonitor

def cfg_run(**kwargs): 
    with open("{}.yaml".format(kwargs['output']), 'w', encoding='utf8') as file:
        yaml.dump(kwargs, file, default_flow_style=False, allow_unicode=True)
    del kwargs['cores']
    run(**kwargs)

def run(cfg, num_timesteps, seed, hid_size, **kwargs):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.configure(dir_path, ['stdout', 'log'])
    
    sess = U.single_threaded_session()
    sess.__enter__()
    
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
        
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = GRLEnv(cfg)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=hid_size, num_hid_layers=2)
    env = MyMonitor(env, osp.join(logger.get_dir(), kwargs['output']), report='learn')
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    if kwargs['evaluation']:
        trpo_mpi.play(sess, env, policy_fn, timesteps_per_batch=1024, load_file=kwargs['load_file'])
    else:
        trpo_mpi.learn(sess, env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, **kwargs)
            
    env.close()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--cfg', type=str, default='cfg/rbdl_py_balancing_inf.yaml')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--hid-size', help='Number of neurons in single layer', type=int, default=32)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--evaluation', default=False)
    parser.add_argument('--output', type=str, default='default')
    parser.add_argument('--load-file', type=str, default='')
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg_run(**args)
