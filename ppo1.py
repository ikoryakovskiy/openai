#!/usr/bin/env python3
from grlgym.envs.grl import GRLEnv
from baselines.common import set_global_seeds, tf_util as U
import gym, logging
from baselines import logger
import yaml
from my_monitor import MyMonitor
import os.path as osp
from baselines.ppo1 import mlp_policy, pposgd_simple

def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser."""
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)

def cfg_run(**kwargs):
    with open("{}.yaml".format(kwargs['output']), 'w', encoding='utf8') as file:
        yaml.dump(kwargs, file, default_flow_style=False, allow_unicode=True)
    del kwargs['cores']
    run(**kwargs)

def run(cfg, num_timesteps, seed, timesteps_per_actorbatch, **kwargs):
    dir_path = osp.dirname(osp.realpath(__file__))
    logger.configure(dir_path, ['stdout'])

    #U.make_session(num_cpu=1).__enter__()
    sess = U.single_threaded_session()
    sess.__enter__()

    set_global_seeds(seed)
    env = GRLEnv(cfg)
    env.set_test(False)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = MyMonitor(env, osp.join(logger.get_dir(), kwargs['output']), report='learn')
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    if kwargs['evaluation']:
        pposgd_simple.play(sess, env, policy_fn, timesteps_per_actorbatch=1024, load_file=kwargs['load_file'])
    else:
        pposgd_simple.learn(sess, env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=timesteps_per_actorbatch,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear', **kwargs
            )
    env.close()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--cfg', type=str, default='cfg/rbdl_py_balancing_inf.yaml')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--timesteps-per-actorbatch', type=int, default=int(2048))
    boolean_flag(parser, 'evaluation', default=False)
    parser.add_argument('--output', type=str, default='default')
    parser.add_argument('--load-file', type=str, default='')
    boolean_flag(parser, 'save', default=False)
    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg_run(**args)
