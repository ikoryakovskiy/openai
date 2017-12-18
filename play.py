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
args['eval_cfg'] = 'cfg/rbdl_py_{}_play.yaml'.format(task)
args['layers_shape'] = '400, 300'
args['nb_timesteps'] = None
args['nb_trials'] = 1
args['test_interval'] = 0
args['noise_type'] = 'ou_0.15_0.20'
args['normalize_observations'] = False
args['load_file'] = 'rbdl_py_{}'.format(task)
args['output'] = 'rbdl_py_{}_play'.format(task)
args['save'] = False

# Run actual script.
run(**args)