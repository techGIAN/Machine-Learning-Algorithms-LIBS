# run this first before running the models
# Usage: python3 preprocess.py --dataset [path_to_dataset] --seed [positive_int_seed]

#imports
import pandas as pd
import os
import argparse
from utilities.DataSplitter import DataSplitter


# =============================== Helper Functions ===============================

def check_positive(v):
    int_val = int(v)
    if int_val <= 0:
        raise argparse.ArgumentTypeError('Invalid seed.')
    return int_val

def get_args():
    '''
        Gets the arguments
    '''
    parser = argparse.ArgumentParser(description='Parameters set for the models.')
    parser.add_argument('--dataset', type=str, default='../dataset/spec_df.csv', help='The spectroscopic dataset.')
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
df_name = args.dataset
seed = args.seed

path = '../dataset/'
kind = ['validation', 'testing', 'ensembling']
path_list = [path + x for x in ['train_set/', 'val_set/', 'test_set/', 'ensemble_set/']]
for p in path_list:
    if not os.path.exists(p):
        os.mkdir(p)

df = pd.read_csv(df_name)
data_splitter = DataSplitter(seed=seed)
partitioned_data = data_splitter.split_data(df)

for i in range(5):
    partitioned_data[i].to_csv(path_list[0] + 'training_fold_' + str(i+1) + '.csv', index=False)


for i in range(5,8):
    partitioned_data[i].to_csv(path_list[i-4] + kind[i-5] + '.csv', index=False)

print('Data preprocess complete.')