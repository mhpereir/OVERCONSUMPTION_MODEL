# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:33:14 2020

@author: Matthew Wilson
"""
import os, argparse, json

with open("params.json") as paramfile:
    params = json.load(paramfile)
    print(params['model_setup']['n_galaxy'])
