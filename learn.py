#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:14:17 2017

@author: ivan
"""
import os

cfg = 'cfg/rbdl_py_balancing.yaml'
nb_epochs = 2
output = os.path.splitext(os.path.basename(cfg))[0]

args = '--cfg={} --nb-epochs={} --output={}'.format(cfg, nb_epochs, output)
os.system('python3 ddpg.py %s' % args)
