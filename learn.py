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
args['layers_shape'] = '400, 300'
args['nb_timesteps'] = 300000
args['test_interval'] = 30
args['noise_type'] = 'ou_0.15_0.20'
args['normalize_observations'] = False
args['output'] = 'rbdl_py_{}'.format(task)
args['save'] = True


'''
import yaml
with open('tmp/cfg_rbdl_py_walking-30000000-000000-000000-000000-000000-000100-mp0.yaml', 'r') as file:
    args = yaml.load(file)
'''

# Run actual script.
run(**args)
