#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
import os

cfg = 'cfg/rbdl_py_balancing_play.yaml'
nb_epochs = 1
nb_epoch_cycles = 0
nb_rollout_steps = 0
output = ''
load_file = 'rbdl_py_balancing'

args = '--cfg={} --nb-epochs={} --nb-epoch-cycles={} --nb-rollout-steps={} --output={} --load-file={}'.format(
    cfg, nb_epochs, nb_epoch_cycles, nb_rollout_steps, output, load_file)
os.system('python3 ddpg.py %s' % args)