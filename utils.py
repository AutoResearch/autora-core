#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### utilities file for Bayesian Scientist / parallel machine scientist (pms)
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

from mcmc import *
from parallel import *

from copy import deepcopy
from ipywidgets import IntProgress
from IPython.display import display

import matplotlib.pyplot as plt


def run(pms,
        num_steps,
        equilibration_margin=1,
        thinning=100,
        anneal=False,
        clear_cache=True):
    prog_bar = init_prog(num_steps) # make progress bar
    desc_len, model, model_len = [], None, np.inf
    for n in range(num_steps):
        step(pms,prog_bar)
        if(num_steps % thinning == 0): # sample less often if we thin more
            desc_len.append(pms.t1.E)  # Add the description length to the trace
        if pms.t1.E < model_len:  # Check if this is the MDL expression so far
            model, model_len = deepcopy(pms.t1), pms.t1.E
    return model, model_len, desc_len

def step(pms,prog_bar):
    # MCMC update
    pms.mcmc_step() # MCMC step within each T
    pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
    prog_bar.value += 1
    display(prog_bar)
    return 0

def init_prog(num_steps):
    # Draw a progress bar to keep track of the MCMC progress
    prog_bar = IntProgress(min=0, max=num_steps, description='Running:') # instantiate the bar
    display(prog_bar)
    return prog_bar

def present_results(pms,
                    model,
                    model_len,
                    desc_len):
    print('Best model:\t', model)
    print('Desc. length:\t', model_len)
    plt.figure(figsize=(15, 5))
    plt.plot(desc_len)
    plt.xlabel('MCMC step', fontsize=14)
    plt.ylabel('Description length', fontsize=14)
    plt.title('MDL model: $%s$' % model.latex())
    plt.show()
    return 0

# BUG
def predict(model,x,y):
    plt.figure(figsize=(6, 6))
    plt.scatter(model.predict(x),y) # BUG: 
    plt.plot((-6, 0), (-6, 0))
    plt.xlabel('MDL model predictions', fontsize=14)
    plt.ylabel('Actual values', fontsize=14)
    plt.show()