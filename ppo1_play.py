#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from ppo1 import parse_args, cfg_run

args = parse_args()

task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}_inf_play.yaml'.format(task)
args['evaluation'] = True
args['output'] = 'ppo1-rbdl_py_{}_play'.format(task)
args['load_file'] = 'ppo1-rbdl_py_{}_inf'.format(task)

    
'''
import yaml
with open('tmp/ddpg-cfg_rbdl_py_balancing-10000000-000000-000000-000000-000000-000100-000000-mp0.yaml', 'r') as file:
    args = yaml.load(file)
'''

# Run actual script.
cfg_run(**args)
