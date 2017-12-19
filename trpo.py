#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from grlgym.envs.grl import GRLEnv

from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import logging
import gym
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import baselines.common.tf_util as U

from my_monitor import MyMonitor

def train(cfg, num_timesteps, seed, output):
    
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
            hid_size=32, num_hid_layers=2)
    env = MyMonitor(env, osp.join(logger.get_dir(), output))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, default='cfg/rbdl_py_walking.yaml')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--output', type=str, default='default')
    args = parser.parse_args()
    logger.configure()
    train(args.cfg, num_timesteps=args.num_timesteps, seed=args.seed, output=args.output)


if __name__ == '__main__':
    main()
