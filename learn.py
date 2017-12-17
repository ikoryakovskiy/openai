#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from ddpg import parse_args, run
import yaml

args = parse_args()

'''
args['cfg'] = 'cfg/rbdl_py_balancing.yaml'
args['eval_cfg'] = 'cfg/rbdl_py_balancing.yaml'
args['nb_timesteps'] = 100000
args['test_interval'] = 30
args['noise_type'] = 'ou_0.15_0.20'
args['normalize_observations'] = False
args['output'] = 'rbdl_py_balancing'
'''

with open('tmp/cfg_rbdl_py_walking-30000000-000000-000000-000000-000000-000100-mp0.yaml', 'r') as file:
    args = yaml.load(file)

# Run actual script.
run(**args)
