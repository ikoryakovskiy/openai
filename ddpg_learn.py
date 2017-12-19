#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from ddpg import parse_args, run

args = parse_args()

task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}.yaml'.format(task)
args['eval_cfg'] = 'cfg/rbdl_py_{}.yaml'.format(task)
args['architecture'] = 'Divyam'
args['nb_timesteps'] = 300000
args['test_interval'] = 30
args['noise_type'] = 'ou_0.15_0.20'
args['critic_l2_reg']= 0.001
args['normalize_observations'] = True
args['normalize_returns'] = True
args['layer_norm'] = True
args['output'] = 'rbdl_py_{}'.format(task)
args['save'] = True


'''
import yaml
with open('tmp/ddpg-cfg_rbdl_py_balancing-10000000-000000-000000-000000-000000-000100-000000-mp0.yaml', 'r') as file:
    args = yaml.load(file)
'''

# Run actual script.
run(**args)
