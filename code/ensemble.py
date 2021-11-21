import os
import numpy as np
import argparse
import copy
# import shutil

# =============================== Helper Functions ===============================

def get_args():
    '''
        Gets the arguments
    '''
    elements = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
    parser = argparse.ArgumentParser(description='Parameters set for the models.')
    parser.add_argument('--pls', type=int, help='The out id for PLS.')
    parser.add_argument('--svr', type=int, help='The out id for SVR.')
    parser.add_argument('--linreg', type=int, help='The out id for Linear Regression.')
    parser.add_argument('--ridge', type=int, help='The out id for Ridge Regression.')
    parser.add_argument('--lasso', type=int, help='The out id for Lasso Regression.')
    parser.add_argument('--enets', type=int, help='The out id for Elastic Nets Regression.')
    parser.add_argument('--ann', type=int, help='The out id for ANN.')
    parser.add_argument('--cnn', type=int, help='The out id for CNN.')
    parser.add_argument('--xgb', type=int, help='The out id for XGBoost.')
    parser.add_argument('--element', type=str, choices=elements, default='SiO2', help='The element.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        print('Check the parameters set.')
        exit()
    
    return args

def triple_zero_pad(x):
    return '0'*(3-len(str(x))) + str(x)

def get_rmse_lists(directory, model_start_id_dict, element):
    filenames = []
    models = ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'ann', 'cnn', 'xgb']

    for model in models:
        dirt = directory + model + '/'
        fname = 'out' + model_start_id_dict[model] + '_element=' + element + '_model=' + model + '.txt'
        filenames.append(dirt + fname)

    rmsecs = []
    rmsevs = []
    rmsets = []
    rmsees = []

    for fn in filenames:
        counter = 0
        with open(fn) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if 'RMSEC' in line:
                    arr = line.split(':')
                    rmsecs.append(float(arr[1]))
                    counter += 1
                if 'RMSEV' in line:
                    arr = line.split(':')
                    rmsevs.append(float(arr[1]))
                    counter += 1
                if 'RMSET' in line:
                    arr = line.split(':')
                    rmsets.append(float(arr[1]))
                    counter += 1
                if 'RMSEE' in line:
                    arr = line.split(':')
                    rmsees.append(float(arr[1]))
                    counter += 1
                if counter > 3:
                    break
                line = f.readline()
    
    return (rmsecs, rmsevs, rmsets, rmsees)

def get_output_file(model, element, r_type, f_type, number=None):
    '''
        Returns the output filename
    '''
    out_dir = '../results/' + model
    plot = 'plot_out' if f_type == 'png' else 'out'
    files = [fi for fi in listdir(out_dir) if isfile(join(out_dir, fi))]
    files = [fi[3:] for fi in files if fi[0:3] == 'out']
    nums = [-1] if len(files) == 0 else []
    for fi in files:
        string_dig = ''
        i = 1
        while fi[0:i].isdigit():
            string_dig = fi[0:i]
            i += 1
        nums.append(int(string_dig))
    max_num = max(nums)+1 if plot == 'out' else max(nums)
    str_max_num = str(max_num)
    str_num = '0'*(3-len(str_max_num)) + str_max_num if number is None else number
    f_out = out_dir + '/' + plot + str_num + '_element=' + element + '_' + r_type + '=' + model + '.' + f_type
    return f_out

def get_top_three(l):
    temp_l = l.copy()
    min_score = min(temp_l)
    first_place = temp_l.index(min_score)

    temp_l[first_place] = np.inf
    min_score = min(temp_l)
    second_place = temp_l.index(min_score)

    temp_l[second_place] = np.inf
    min_score = min(temp_l)
    third_place = temp_l.index(min_score)

    return (first_place, second_place, third_place)

def scores(rmsecs, rmsecvs):
    rmsec_arr = np.array(rmsecs)
    rmsecv_arr = np.array(rmsecvs)
    deviations = rmsec_arr - rmsecv_arr
    abs_deviations = abs(deviations)
    scores = 0.5*abs_deviations + 0.3*deviations + 0.1*rmsec_arr + 0.1*rmsecv_arr
    return [round(s, 4) for s in scores]


# =============================== Main Function ===============================

args = get_args()
pls_start = args.pls
svr_start = args.svr
linreg_start = args.linreg
ridge_start = args.ridge
lasso_start = args.lasso
enets_start = args.enets
ann_start = args.ann
cnn_start = args.cnn
xgb_start = args.xgb
element = args.element

starts = [pls_start, svr_start, linreg_start, ridge_start, lasso_start, enets_start, ann_start, cnn_start, xgb_start]
starts_id = [triple_zero_pad(x) for x in starts]
models = ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'ann', 'cnn', 'xgb']

model_start_id_dict = dict(zip(models, starts_id))
directory = '../results/'

rmsecs, rmsevs, rmsets, rmsees = get_rmse_lists(directory, model_start_id_dict, element)


new_scores = scores(rmsecs, rmsets)
model_1, model_2, model_3 = get_top_three(new_scores)
ensemble_res = round(np.mean([rmsees[model_1], rmsees[model_2], rmsees[model_3]]), 4)

f = open(directory + 'ensemble_result.txt', 'w')

s = 'FINAL COMPARISON OF RESULTS with ENSEMBLE (' + element + ')          ' 
temp_s = s
s = '='*len(s) + '\n' + s + '\n' + '='*len(s) + '\n'

headings = ['MODEL\t', 'RMSEC', 'RMSEV', 'RMSET', 'RMSEE']
for h in headings:
    s += h + '\t'
s += '\n' + '-----' + '\t\t' + '-----' + '\t' + '-----' + '\t' + '-----' + '\t' + '-----' + '\n'

dict_place = {model_1: 1, model_2: 2, model_3:3 }
for i in range(9):
    if i in dict_place.keys():
        s += models[i] + '\t\t' + str(rmsecs[i]) + '\t' + str(rmsevs[i]) + '\t' + str(rmsets[i]) + '\t' + str(rmsees[i]) + ' (' + str(dict_place[i]) + ')' + '\n'
    else:
        s += models[i] + '\t\t' + str(rmsecs[i]) + '\t' + str(rmsevs[i]) + '\t' + str(rmsets[i]) + '\t' + str(rmsees[i]) + '\n'

s += 'ensemble' + '\t\t' + ' --' + '\t' + ' --' + '\t' + ' --' + '\t' + str(ensemble_res) + '\n'

s += '-'*len(temp_s) + '\n'
s += 'Models for Ensemble:' + '\n'
s += '\t1) ' + models[model_1] + '\n'
s += '\t2) ' + models[model_2] + '\n'
s += '\t3) ' + models[model_3] + '\n'

s += '='*len(temp_s)

f.write(s)

f.close()

print('Successful ensembling.')