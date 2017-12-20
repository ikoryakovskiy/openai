#!/usr/bin/env python3
from grlgym.envs.grl import GRLEnv
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import yaml

def cfg_run(**kwargs): 
    with open("{}.yaml".format(kwargs['output']), 'w', encoding='utf8') as file:
        yaml.dump(kwargs, file, default_flow_style=False, allow_unicode=True)
    del kwargs['cores']
    run(kwargs)

def run(cfg, num_timesteps, seed, **kwargs):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    
    #U.make_session(num_cpu=1).__enter__()
    sess = U.single_threaded_session()
    sess.__enter__()
    
    set_global_seeds(seed)
    env = GRLEnv(cfg)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    if kwargs['evaluation']:
        trpo_mpi.play(sess, env, policy_fn, timesteps_per_batch=1024, load_file=kwargs['load_file'])
    else:
        pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=2048,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear',
            )
    env.close()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--cfg', type=str, default='cfg/rbdl_py_balancing_inf.yaml')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--evaluation', default=True)
    parser.add_argument('--output', type=str, default='default')
    parser.add_argument('--load-file', type=str, default='')
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg_run(**args)
