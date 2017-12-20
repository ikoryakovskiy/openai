#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from ddpg import parse_args, cfg_run

args = parse_args()

task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}_play.yaml'.format(task)
args['architecture'] = 'Divyam'
args['num_timesteps'] = None
args['nb_trials'] = 1
args['test_interval'] = 0
args['noise_type'] = 'ou_0.15_0.20'
args['normalize_observations'] = True
args['load_file'] = 'rbdl_py_{}'.format(task)
args['output'] = 'rbdl_py_{}_play'.format(task)
args['save'] = False

# Run actual script.
cfg_run(**args)