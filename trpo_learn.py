#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
from trpo import parse_args, cfg_run
import os


'''
args = parse_args()

task = 'walking'
#task = 'balancing'

args['cfg'] = 'cfg/rbdl_py_{}_inf.yaml'.format(task)
#args['architecture'] = 'Divyam'
args['num_timesteps'] = 1000
args['evaluation'] = False
args['output'] = 'trpo-rbdl_py_{}_inf'.format(task)
args['load_file'] = ''

'''

import yaml
with open('tmp/trpo-cfg_rbdl_py_walking_inf-6400000-100000000-mp0.yaml', 'r') as file:
    di = yaml.load(file)

def dict_to_args(di):
    args = ""
    for key in di:
        if not di[key] == '':
            new_key = key.replace('_', '-')
            if type(di[key]) is not bool:
                args += '--' + new_key + '=' + str(di[key]) + ' '
            elif di[key] == True:
                args += '--' + new_key + ' '
            else:
                args += '--no-' + new_key + ' '
    return args

args = dict_to_args(di)
print(args)
os.system("python trpo.py {}".format(args))

'''
# Run actual script.
args['save'] = True
cfg_run(**args)
'''