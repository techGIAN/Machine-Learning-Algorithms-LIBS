import os
import pandas as pd
import numpy as np
import argparse
import copy
import matplotlib.pyplot as plt
import warnings

from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn import linear_model as lm
import xgboost as xgb

from sklearn.metrics import mean_squared_error


# =============================== Helper Functions ===============================

def get_args():
    '''
        Gets the arguments
    '''
    elements = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
    parser = argparse.ArgumentParser(description='Parameters set for the models.')
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

def isnum(x):
    try:
        float(x)
    except:
        return False
    else:
        return True

def get_params(filename):
    param_dict = dict()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if 'params' in line:
                line = line.replace('params = ','')
                line = line[1:-1]
                arr = line.split(', ')
                for a in arr:
                    ar =  a.split(': ')
                    ar[0] = ar[0][1:-1]
                    ar[1] = float(ar[1]) if isnum(ar[1]) else ar[1][1:-1]
                    ar[1] = int(ar[1]) if isnum(ar[1]) and int(ar[1]) == ar[1] else ar[1]
                    param_dict[ar[0]] = ar[1]
    if 'model_id' in param_dict:
        param_dict.pop('model_id')
    return param_dict

def test_plot(actual_y, predicted_y1, predicted_y2, predicted_y3, models, element, fname):
    '''
        Plot the Actual vs Predicted Y (only for the validation or testing plot)
    '''
    actual_values = actual_y.tolist()
    predicted_values1 = predicted_y1.tolist()
    predicted_values2 = predicted_y2.tolist()
    predicted_values3 = predicted_y3.tolist()
    pred_values = [predicted_values1, predicted_values2, predicted_values3]
    min_actual_y = min(actual_values)
    max_actual_y = max(actual_values) + 1
    colors = ['r', 'b', 'g']
    
    plt.figure(figsize=(11,3))
    for j in range(3):
        plt.subplot(1,3,j+1)
        x_label = 'Actual ' + element
        y_label = 'Predicted ' + element

        # Plot y = x
        x_range = np.arange(min_actual_y, max_actual_y)
        y_range = x_range
        plt.plot(x_range, y_range, color='k', linestyle='-')

        # scatter plot of the points
        plt.scatter(actual_y, pred_values[j], s=5, color=colors[j])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.title(title)

    plt.suptitle('Actual vs Predicted Plots for ' + element)
    plt.tight_layout()

    plt.savefig(fname)
    plt.show()


# =============================== Main Function ===============================

np.random.seed(123)
args = get_args()
element = args.element

elements = {'SiO2': 1,
            'TiO2': 2,
            'Al2O3': 3,
            'FeOT': 4,
            'MgO': 5,
            'CaO': 6,
            'Na2O': 7,
            'K2O': 8}


directory = '../spectroscopy-preliminary-results/' + element + '/'


ens_df = pd.read_csv('../dataset/ensemble_set/ensembling.csv')
ens_actuals = ens_df.iloc[:,elements[element] + 6144]
ens_predictors = ens_df.iloc[:,1:6145]
ens_predictors = ens_predictors.subtract(ens_predictors.mean())

no_ens_df = pd.read_csv('../dataset/without_ensemble/non_ensemble.csv')
no_ens_actuals = no_ens_df.iloc[:,elements[element] + 6144]
no_ens_predictors = no_ens_df.iloc[:,1:6145]
no_ens_predictors = no_ens_predictors.subtract(ens_predictors.mean())

final_output = ''
models_output = ''
models_for_ensemble = []
warnings.filterwarnings('ignore')
fl = True
with open(directory + 'ensemble_result.txt') as f:
    for line in f:
        if fl:
            temp_line = line
            fl = False
        if 'ensemble' in line:
            continue
        if line == 'Models for Ensemble:\n':
            models_output = line
            for i in range(1,4):
                line = f.readline()
                models_output += line
                line = line.strip()
                models_for_ensemble.append(line.split(' ')[1])
            break
        else:
            final_output += line

ens_preds_ave = [0] * ens_actuals.shape[0]

if True:
    directory_model = directory + 'pls'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = PLSRegression(n_components = model_param_dict['n_components'])
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    if 'pls' in models_for_ensemble:
        ens_preds_ave = ens_preds_ave + preds
    pls_preds = copy.deepcopy(preds)

if 'svr' in models_for_ensemble:
    directory_model = directory + 'svr'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = SVR()
    ens_model.set_params(**model_param_dict)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    ens_preds_ave = ens_preds_ave + preds

if 'linreg' in models_for_ensemble:
    directory_model = directory + 'linreg'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = lm.LinearRegression(fit_intercept=True)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    ens_preds_ave = ens_preds_ave + preds

if 'lasso' in models_for_ensemble:
    directory_model = directory + 'lasso'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = lm.Lasso()
    ens_model.set_params(**model_param_dict)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    ens_preds_ave = ens_preds_ave + preds

if 'ridge' in models_for_ensemble:
    directory_model = directory + 'ridge'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    model_param_dict['solver'] = 'svd'
    ens_model = lm.Ridge()
    ens_model.set_params(**model_param_dict)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    ens_preds_ave = ens_preds_ave + preds

if 'enets' in models_for_ensemble:
    directory_model = directory + 'enets'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = lm.ElasticNet()
    ens_model.set_params(**model_param_dict)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    ens_preds_ave = ens_preds_ave + preds

if True:
    directory_model = directory + 'xgb'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = xgb.XGBRegressor()
    ens_model.set_params(**model_param_dict)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    if 'xgb' in models_for_ensemble:
        ens_preds_ave = ens_preds_ave + preds
    xgb_preds = copy.deepcopy(preds)

if 'ann' in models_for_ensemble:
    directory_model = directory + 'ann'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = ANN()
    ens_model.set_params(**model_param_dict)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    ens_preds_ave = ens_preds_ave + preds

if 'cnn' in models_for_ensemble:
    directory_model = directory + 'cnn'
    param_file = [x for x in os.listdir(directory_model) if x[0] != '.' and 'params' in x][0]
    model_param_dict = get_params(directory_model + '/' + param_file)
    ens_model = CNN()
    ens_model.set_params(**model_param_dict)
    ens_model.fit(no_ens_predictors, no_ens_actuals)
    preds = ens_model.predict(ens_predictors).reshape((ens_actuals.shape[0],))
    ens_preds_ave = ens_preds_ave + preds

ens_preds_ave = ens_preds_ave/3
test_plot(ens_actuals, pls_preds, xgb_preds, ens_preds_ave, ['PLS', 'XGB', 'Ensemble'], element, '../spectroscopy-preliminary-results/' + element + '/actual-predicted-plot.png')
