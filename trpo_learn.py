#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from trpo import parse_args, cfg_run

args = parse_args()
'''
task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}_inf.yaml'.format(task)
#args['architecture'] = 'Divyam'
args['num_timesteps'] = 2000000
args['evaluation'] = False
args['output'] = 'rbdl_py_{}_inf'.format(task)
args['load_file'] = ''
args['save'] = True
'''
    

import yaml
with open('tmp/trpo-cfg_rbdl_py_walking_inf-6400000-100000000-mp0.yaml', 'r') as file:
    args = yaml.load(file)


# Run actual script.
cfg_run(**args)
