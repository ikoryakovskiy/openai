#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from ppo1 import parse_args, cfg_run

'''
args = parse_args()

task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}_inf.yaml'.format(task)
#args['architecture'] = 'Divyam'
args['num_timesteps'] = 1000
args['evaluation'] = False
args['output'] = 'ppo1-rbdl_py_{}_inf'.format(task)
args['load_file'] = ''

    
'''
import yaml
with open('tmp/ppo1-cfg_rbdl_py_balancing_inf-102400000-20000000-mp0.yaml', 'r') as file:
    args = yaml.load(file)


# Run actual script.
args['save'] = True
cfg_run(**args)
