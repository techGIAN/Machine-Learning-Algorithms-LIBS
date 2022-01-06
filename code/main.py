# Main model to use only if you are not using Neural Networks.
# 

# imports
import pandas as pd
import numpy as np
from datetime import datetime as dt
from numpy.random import seed
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import argparse
import os

from models.PLSModel import PLSModel
from models.SVRModel import SVRModel
from models.LRMModel import LRMModel
from models.XGBModel import XGBModel




# =============================== Helper Functions ===============================

def check_positive(v):
    int_val = int(v)
    if int_val <= 0:
        raise argparse.ArgumentTypeError('Invalid seed.')
    return int_val

def read_params(filename):
    '''
        Given the parameter dictionary txt file, read it.
    '''
    params = dict()
    with open(filename) as f:
        line = f.readline()
        while line:
            arr = line.split(':')
            arr[0] = eval(arr[0].strip())
            arr[1] = eval(arr[1].strip())
            params[arr[0]] = arr[1]
            line = f.readline()
        
    return params

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

def create_result_folder(models):
    '''
        Create results folder if they do not exist
    '''
    directory = '../results/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    for m in models:
        if not os.path.exists(directory + m):
            os.mkdir(directory + m)

def get_args():
    '''
        Gets the arguments
    '''
    model_choices = ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'xgb']
    elements = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
    parser = argparse.ArgumentParser(description='Parameters set for the models.')
    parser.add_argument('--model', type=str, choices=model_choices, help='The model.')
    parser.add_argument('--params', type=str, help='The parameters used for model as a list.')
    parser.add_argument('--element', type=str, choices=elements, default='SiO2', help='The element.')
    parser.add_argument('--seed', type=check_positive, default=123, help='The seed.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        print('Check the parameters set.')
        exit()
    
    return args

# =============================== Main Function ===============================

args = get_args()
model = args.model
params = args.params
E = args.element
seed = args.seed

# check if params exists
if not os.path.exists(params):
    print('Parameter file does not exist.')
    exit()
else:
    param_dict = read_params(params)

# create the results folder if it does not exist.
model_choices = ['pls', 'svr', 'linreg', 'ridge', 'lasso', 'enets', 'xgb']
create_result_folder(model_choices)

# get the training_set
train_dir = '../dataset/train_set/'
files = [train_dir + fi.replace('._', '') for fi in listdir(train_dir) if isfile(join(train_dir, fi))]
df_tuple = tuple()
for f in files:
    df = pd.read_csv(f)
    df_tuple += (df,)

df_val_name = '../dataset/val_set/validation.csv'
df_test_name = '../dataset/test_set/testing.csv'
df_ens_name = '../dataset/ensemble_set/ensembling.csv'

df_val = pd.read_csv(df_val_name)
df_test = pd.read_csv(df_test_name)
df_ens = pd.read_csv(df_ens_name)


t1 = dt.now()
print('Start runtime: ', t1)
print('Running...')


elements = {'SiO2': 1,
            'TiO2': 2,
            'Al2O3': 3,
            'FeOT': 4,
            'MgO': 5,
            'CaO': 6,
            'Na2O': 7,
            'K2O': 8}

ele = elements[E]

N_predictors = 6144
target_idx = N_predictors + ele



# =============================== PLS ===============================

if model == 'pls':

    pls_reg = PLSModel(seed=seed)
    pls_bounds = param_dict['component_bound']
    opt_dict, min_rmsec, pls_models, rmsecs, rmsecvs, n_comp = pls_reg.pls_train(df_tuple, df_val, (1,6145), target_idx,
                                                                component_bound=np.arange(int(pls_bounds[0]),
                                                                                            int(pls_bounds[1])))
    rmsets = pls_reg.pls_test(df_test, (1,6145), target_idx, n_comp, pls_models, test_plot=False)
    rmsees = pls_reg.pls_test(df_ens, (1,6145), target_idx, n_comp, pls_models, test_plot=False)

    f_out = get_output_file(model, E, 'model', 'txt')

    f = open(f_out, 'w')
    f.write('PLS for ' + E + '\n' + '='*len('PLS for ' + E) + '\n')
    f.write('Best number of components: ' + str(n_comp) + '\n')
    f.write('\tRMSEC: ' + str(min_rmsec) + '\n')
    f.write('\tRMSEV: ' + str(rmsecvs[opt_dict['index']]) + '\n')
    f.write('\tRMSET: ' + str(rmsets[opt_dict['index']]) + '\n')
    f.write('\tRMSEE: ' + str(rmsees[opt_dict['index']]) + '\n')
    f.write('\n')
    for i in range(len(rmsecs)):
        f.write('Component ' + str(i+2) + ': \n')
        f.write('\tRMSEC: ' + str(rmsecs[i]) + '\n')
        f.write('\tRMSEV: ' + str(rmsecvs[i]) + '\n')
        f.write('\tRMSET: ' + str(rmsets[i]) + '\n')
        f.write('\tRMSEE: ' + str(rmsees[i]) + '\n')

    f.write('\n')

    print('Done training PLS with element ' + E + '.')

    t2 = dt.now()
    print('End runtime: ', t2)

    time_total = 'Time it took: ' + str(t2-t1)
    print(time_total)

    f.write(time_total)

    f.close()

    fout_arr = f_out.split('_')
    N = fout_arr[0][-3:]
    f_out = get_output_file(model, E, 'out_params', 'txt', number=N)
    f = open(f_out, 'w')

    opt_dict['rmset'] = rmsets[opt_dict['index']]
    opt_dict['rmsee'] = rmsees[opt_dict['index']]
    f.write('Best model parameters for element ' + E + ' for model PLS:' + '\n')
    for k, v in opt_dict.items():
        f.write('\t' + k + ' = ' + str(v) + '\n')
    f.close()


# =============================== SVR ===============================

elif model == 'svr':

    svr_reg = SVRModel(seed=seed)
    svr_params = param_dict

    opt_dict, opt_rmsec, svr_models, rmsecs, rmsecvs, parameter_dictionaries = svr_reg.svr_train(df_tuple, df_val, (1,6145), target_idx, svr_params, dim_reduce=0)
    # min_rmsec, svr_models, rmsecs, opt_params, dim_reduced_instances, parameter_dictionaries = svr_reg.svr_train(df_tuple, target_idx, svr_params, dim_reduce=0)
    rmsets = svr_reg.svr_test(df_test, (1,6145), target_idx, svr_models, parameter_dictionaries, test_plot=False)
    rmsees = svr_reg.svr_test(df_ens, (1,6145), target_idx, svr_models, parameter_dictionaries, test_plot=False)

    f_out = get_output_file(model, E, 'model', 'txt')

    f = open(f_out, 'w')
    f.write('SVR for ' + E + '\n' + '='*len('SVR for ' + E) + '\n')
    f.write('Best Model ID: ' + str(opt_dict['params']['model_id']) + '\n')
    f.write('\tRMSEC: ' + str(opt_rmsec) + '\n')
    f.write('\tRMSEV: ' + str(rmsecvs[opt_dict['index']]) + '\n')
    f.write('\tRMSET: ' + str(rmsets[opt_dict['index']]) + '\n')
    f.write('\tRMSEE: ' + str(rmsees[opt_dict['index']]) + '\n')
    f.write('\n')
    for i in range(len(rmsecs)):
        f.write('Model ' + str(i) + ': \n')
        f.write('\tRMSEC: ' + str(rmsecs[i]) + '\n')
        f.write('\tRMSEV: ' + str(rmsecvs[i]) + '\n')
        f.write('\tRMSET: ' + str(rmsets[i]) + '\n')
        f.write('\tRMSEE: ' + str(rmsees[i]) + '\n')

    f.write('\n')

    print('Done training SVR with element ' + E + '.')

    t2 = dt.now()
    print('End runtime: ', t2)

    time_total = 'Time it took: ' + str(t2-t1)
    print(time_total)

    f.write(time_total)

    f.close()

    fout_arr = f_out.split('_')
    N = fout_arr[0][-3:]
    f_out = get_output_file(model, E, 'out_params', 'txt', number=N)
    f = open(f_out, 'w')

    opt_dict['rmset'] = rmsets[opt_dict['index']]
    opt_dict['rmsee'] = rmsees[opt_dict['index']]
    f.write('Best model parameters for element ' + E + ' for model SVR:' + '\n')
    for k, v in opt_dict.items():
        f.write('\t' + k + ' = ' + str(v) + '\n')
    f.close()


# =============================== LRM ===============================

elif model in ['linreg', 'lasso', 'ridge', 'enets']:

    # lrm_names = [x + ' Regression' for x in ['Linear', 'Lasso', 'Ridge', 'Elastic Net']]
    # lrm_shortnames = dict(zip(lrm_names, ['linreg', 'lasso', 'ridge', 'enets']))               
    # lrm_reg = LRMModel(regularization=lrm_shortnames[model], seed=seed)
    lrm_reg = LRMModel(regularization=model, seed=seed)
    lrm_params = param_dict

    if model == 'linreg':
        lrm_params = dict()
    elif model == 'lasso' or model == 'ridge':
        lrm_params.pop('l1_ratio')


    opt_dict, opt_rmsec, lrm_models, rmsecs, rmsecvs, parameter_dictionaries = lrm_reg.lrm_train(df_tuple, df_val, (1,6145), target_idx, lrm_params, dim_reduce=0)
    rmsets = lrm_reg.lrm_test(df_test, (1,6145), target_idx, lrm_models, parameter_dictionaries, test_plot=False)
    rmsees = lrm_reg.lrm_test(df_ens, (1,6145), target_idx, lrm_models, parameter_dictionaries, test_plot=False)

    f_out = get_output_file(model, E, 'model', 'txt')

    f = open(f_out, 'w')
    f.write(model.upper() + ' for ' + E + '\n' + '='*len(model.upper() + ' for ' + E) + '\n')
    f.write('Best Model ID: ' + str(opt_dict['params']['model_id']) + '\n')
    f.write('\tRMSEC: ' + str(opt_rmsec) + '\n')
    f.write('\tRMSEV: ' + str(rmsecvs[opt_dict['index']]) + '\n')
    f.write('\tRMSET: ' + str(rmsets[opt_dict['index']]) + '\n')
    f.write('\tRMSEE: ' + str(rmsees[opt_dict['index']]) + '\n')
    f.write('\n')
    for i in range(len(rmsecs)):
        f.write('Model ' + str(i) + ': \n')
        f.write('\tRMSEC: ' + str(rmsecs[i]) + '\n')
        f.write('\tRMSEV: ' + str(rmsecvs[i]) + '\n')
        f.write('\tRMSET: ' + str(rmsets[i]) + '\n')
        f.write('\tRMSEE: ' + str(rmsees[i]) + '\n')

    f.write('\n')

    print('Done training ' + model.upper() + ' with element ' + E + '.')

    t2 = dt.now()
    print('End runtime: ', t2)

    time_total = 'Time it took: ' + str(t2-t1)
    print(time_total)

    f.write(time_total)

    f.close()

    fout_arr = f_out.split('_')
    N = fout_arr[0][-3:]
    f_out = get_output_file(model, E, 'out_params', 'txt', number=N)
    f = open(f_out, 'w')

    opt_dict['rmset'] = rmsets[opt_dict['index']]
    opt_dict['rmsee'] = rmsees[opt_dict['index']]
    f.write('Best model parameters for element ' + E + ' for model ' + model.upper() + ':' + '\n')
    for k, v in opt_dict.items():
        f.write('\t' + k + ' = ' + str(v) + '\n')
    f.close()



# # =============================== XGB ===============================

elif model == 'xgb':

    xgb_reg = XGBModel(seed=seed)
    xgb_params = param_dict

    opt_dict, opt_rmsec, xgb_models, rmsecs, rmsecvs, parameter_dictionaries = xgb_reg.xgb_train(df_tuple, df_val, (1,6145), target_idx, xgb_params, dim_reduce=0)
    rmsets = xgb_reg.xgb_test(df_test, (1,6145), target_idx, xgb_models, parameter_dictionaries, test_plot=False)
    rmsees = xgb_reg.xgb_test(df_ens, (1,6145), target_idx, xgb_models, parameter_dictionaries, test_plot=False)

    f_out = get_output_file(model, E, 'model', 'txt')

    f = open(f_out, 'w')
    f.write('XGB for ' + E + '\n' + '='*len('XGB for ' + E) + '\n')
    f.write('Best Model ID: ' + str(opt_dict['params']['model_id']) + '\n')
    f.write('\tRMSEC: ' + str(opt_rmsec) + '\n')
    f.write('\tRMSEV: ' + str(rmsecvs[opt_dict['index']]) + '\n')
    f.write('\tRMSET: ' + str(rmsets[opt_dict['index']]) + '\n')
    f.write('\tRMSEE: ' + str(rmsees[opt_dict['index']]) + '\n')
    f.write('\n')
    for i in range(len(rmsecs)):
        f.write('Model ' + str(i) + ': \n')
        f.write('\tRMSEC: ' + str(rmsecs[i]) + '\n')
        f.write('\tRMSEV: ' + str(rmsecvs[i]) + '\n')
        f.write('\tRMSET: ' + str(rmsets[i]) + '\n')
        f.write('\tRMSEE: ' + str(rmsees[i]) + '\n')

    f.write('\n')

    print('Done training XGB with element ' + E + '.')

    t2 = dt.now()
    print('End runtime: ', t2)

    time_total = 'Time it took: ' + str(t2-t1)
    print(time_total)

    f.write(time_total)

    f.close()

    fout_arr = f_out.split('_')
    N = fout_arr[0][-3:]
    f_out = get_output_file(model, E, 'out_params', 'txt', number=N)
    f = open(f_out, 'w')

    opt_dict['rmset'] = rmsets[opt_dict['index']]
    opt_dict['rmsee'] = rmsees[opt_dict['index']]
    f.write('Best model parameters for element ' + E + ' for model XGB:' + '\n')
    for k, v in opt_dict.items():
        f.write('\t' + k + ' = ' + str(v) + '\n')
    f.close()
