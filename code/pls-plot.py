import os
import pandas as pd
import numpy as np
import argparse
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

elements = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
ele_dict = {x:elements.index(x)+1 for x in elements}

all_rmsecs = []
all_rmsecvs = []
best_comps = []

for E in elements:
    directory = '../spectroscopy-preliminary-results/' + E + '/pls/'
    model_file = [x for x in os.listdir(directory) if x[0] != '.' and 'params' not in x][0]
    
    rmsecs = []
    rmsecvs = []

    with open(directory + model_file) as f:
        for line in f:
            if 'Best' in line:
                line = line.strip()
                arr = line.split(': ')
                comps = float(arr[1])

                line = f.readline()
                line = f.readline()
                line = line.strip()
                arr = line.split(': ')
                r = float(arr[1])
        
                best_comps.append((comps,r))
            if 'Component ' in line:
                line = f.readline()
                line = line.strip()
                arr = line.split(': ')
                rmsecs.append(float(arr[1]))

                line = f.readline()
                line = line.strip()
                arr = line.split(': ')
                rmsecvs.append(float(arr[1]))
    
    all_rmsecs.append(rmsecs)
    all_rmsecvs.append(rmsecvs)

max_rmsecvs = [max(L) for L in all_rmsecvs]

plt.figure(figsize=(14,6))
for i in range(1,9):
    plt.subplot(2,4,i)
    x_label = 'n_components for ' + elements[i-1]
    y_label = 'RMSE'

    rng = range(2,51)
    plt.plot(rng, all_rmsecs[i-1], color='darkorange', label='RMSEC')
    plt.plot(rng, all_rmsecvs[i-1], color='purple', label='RMSECV')
    plt.axvline(x=best_comps[i-1][0], color='r', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(max(best_comps[i-1][0]-8,1.5), max_rmsecvs[i-1]*0.95, str(int(best_comps[i-1][0])) + ' components', fontsize = 8, 
         bbox = dict(facecolor = 'mistyrose', alpha = 0.8))

plt.tight_layout()

plt.savefig('../spectroscopy-preliminary-results/pls-plot.png', dpi=200)
plt.show()