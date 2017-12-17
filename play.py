#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from ddpg import parse_args, run

args = parse_args()

args['cfg'] = 'cfg/rbdl_py_balancing.yaml'
args['eval_cfg'] = 'cfg/rbdl_py_balancing_play.yaml'
args['nb_timesteps'] = None
args['nb_trials'] = 1
args['test_interval'] = 0
args['noise_type'] = 'ou_0.15_0.20'
args['normalize_observations'] = False
args['load_file'] = 'rbdl_py_balancing'
args['output'] = 'rbdl_py_balancing_play'

# Run actual script.
run(**args)