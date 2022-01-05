
import os
import numpy as np
import argparse
import shutil

# =============================== Helper Functions ===============================

def get_args():
    '''
        Gets the arguments
    '''
    elements = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
    parser = argparse.ArgumentParser(description='Parameters set for the models.')
    parser.add_argument('--out', type=str, help='The start out.')
    parser.add_argument('--element', type=str, choices=elements, default='SiO2', help='The element.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        print('Check the parameters set.')
        exit()
    
    return args

def scores(rmsecs, rmsecvs):
    rmsec_arr = np.array(rmsecs)
    rmsecv_arr = np.array(rmsecvs)
    deviations = rmsec_arr - rmsecv_arr
    abs_deviations = abs(deviations)
    scores = 0.5*abs_deviations + 0.3*deviations + 0.1*rmsec_arr + 0.1*rmsecv_arr
    return [round(s, 4) for s in scores]

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


# =============================== Main Function ===============================

args = get_args()
out_start = args.out
element = args.element

cnn_dir = '../results/cnn/'
filenames = []
param_files = []

for i in range(1,6):
    out_id = str(int(out_start)+i-1)
    out_id = '0'*(3-len(out_id)) + out_id
    fname = 'out' + out_id + '_element=' + element + '_model=cnn_mod_id=' + str(i) + '.txt'
    filenames.append(cnn_dir + fname)
    p_name = 'out' + out_id + '_element=' + element + '_out_params=cnn_mod_id=' + str(i) + '.txt'
    param_files.append(p_name)

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

final_scores = scores(rmsecs, rmsevs)
min_score = min(final_scores)
opt = final_scores.index(min_score)
opt_model_file = filenames[opt]
opt_param_file = cnn_dir + param_files[opt]

os5 = str(int(out_start) + 5)
os5 = '0'*(3-len(os5)) + os5
new_model_file = cnn_dir + 'out' + os5 + '_element=' + element + '_model=cnn.txt'
new_param_file = cnn_dir + 'out' + os5 + '_element=' + element + '_out_params=cnn.txt'

shutil.copy(opt_model_file, new_model_file)
shutil.copy(opt_param_file, new_param_file)

print('Compile done.')