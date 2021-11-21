# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt

from sklearn.metrics import mean_squared_error

from utilities.Normalizer import Normalizer
from utilities.ModelOptimizer import ModelOptimizer
from sklearn.decomposition import KernelPCA

import xgboost as xgb


import itertools
import warnings

class XGBModel:

    seed = 123
    regulatization = None
    
    def __init__(self, seed=123):
        '''
            Initializes the SVRModel Instance
        '''
        self.seed = seed

    def xgb_param_combos(self, param_dict, N=np.inf):
        '''
            Given a parameter dictionary, generate N combinations.
            Params
            ------
            param_dict: a parameter of dictionaries
            N: number of combos to generate (int) OR
                (float) (0,1) as the percentage of all possible
            Return
            ------
            The list of all combos
        '''
        # check for the validity of N
        if N < 0 or (int(N) != N and N > 1) or N == np.inf:
            print(str(N) + ' is not valid')
            exit()

        # generate a Cartesian product to generate all combinations
        parameter_set = [v for k, v in param_dict.items()]
        param_names = list(param_dict.keys())
        parameter_tuples = list(itertools.product(*parameter_set))

        param_combos = [] # keep all possible combinations to return

        # prune out impossible combination parameters (XGB specific only)
        for param_tup in parameter_tuples.copy():
            params_dict = dict(zip(param_names, param_tup))
            
            # remove all parameters not compatible to gbtree
            # if params_dict['booster'] == 'gbtree':
            for popped_item in ['sample_type', 'normalize_type', 'rate_drop', 'skip_drop']:
                if params_dict.get(popped_item) is not None:
                    params_dict.pop(popped_item)
            
            # if params_dict['tree_method'] != 'approx':
            #     params_dict.pop('sketch_eps')

            # if params_dict['tree_method'] != 'hist' and params_dict['tree_method'] != 'gpu_hist':
            #     for popped_item in ['grow_policy', 'max_bin']:
            #         if params_dict.get(popped_item) is not None:
            #             params_dict.pop(popped_item)
            # else:
            #     if params_dict['grow_policy'] == 'depthwise':
            #         if params_dict.get('max_leaves') is not None:
            #             params_dict.pop('max_leaves')


            # if params_dict['tree_method'] != 'gpu_hist' and params_dict['predictor'] == 'gpu_predictor':
            #     continue
            # else:
            #     
            param_combos.append(params_dict)

        # take a subset if user wants
        if N < np.inf:
            if 0 < N < 1:
                nums = int(N * len(param_combos))
            else:
                nums = N
            param_combos = rnd.sample(param_combos, k=nums)

        counter = 1

        for pc in param_combos.copy():
            pc['model_id'] = counter
            counter += 1
        
        # print('Got the parameters')
        return param_combos


    def xgb_train(self, df_tuple, df_val, predictor_idx, target_idx, params, dim_reduce=-1):


        parameter_combos = self.xgb_param_combos(params, N=100)

        # remove all unnecessary columns
        predictor_tuple = tuple()
        target_tuple = tuple()
        for frame in df_tuple:
            predictor_frame = frame.iloc[:,1:target_idx]
            predictor_tuple += (predictor_frame,)

            target_frame = frame.iloc[:,target_idx]
            target_tuple += (target_frame,)

        df_v_x = df_val.iloc[:,predictor_idx[0]:predictor_idx[1]]
        df_v_y = df_val.iloc[:,target_idx]
        
        # 5 folds
        nums = range(5)
        train_rmsecs = []
        rmsecs = []
        rmsecvs = []
        xgb_models = []

        num_iter = 0

        parameter_dictionaries = []
        m_id = 0

        for param_set in parameter_combos:

            # dict_string = '{'
            # for k, v in param_set.items():
            #     dict_string += k + ': ' + str(v) + ', '
            # dict_string += '}'
            # dict_string = dict_string.replace(', }', '}')
            # print(dict_string)

            # try:
            rmsec = []

            temp_param_set = param_set.copy()
            temp_param_set.pop('model_id')

            xgb_model = xgb.XGBRegressor()
            xgb_model.set_params(**temp_param_set)
            

            for i in nums:

                norm = Normalizer('mean-centering')

                test_x_frame = predictor_tuple[i].reset_index().drop(['index'], axis=1)
                test_y_frame = target_tuple[i].reset_index().drop(['index'], axis=1)

                train_nums = [i for n in nums if i != n]
                trainers_x = [predictor_tuple[tn] for tn in train_nums]
                trainers_y = [target_tuple[tn] for tn in train_nums]
                train_x_frame = trainers_x[0].append([trainers_x[1], trainers_x[2], trainers_x[3]], ignore_index=True)
                train_y_frame = trainers_y[0].append([trainers_y[1], trainers_y[2], trainers_y[3]], ignore_index=True)


                # normalize
                norm_train = norm.normalize(train_x_frame, [0,target_idx-1])
                norm_test = norm.normalize(test_x_frame, [0,target_idx-1])


                # dimensionally reduce if needed
                if dim_reduce > 0 and False:
                    dim_reduced_instance, reduced_train = dr.dim_reduce_train(norm_train, [0, target_idx-1], kern)
                    reduced_test = dr.dim_reduce_test(norm_test, [0,target_idx-1], dim_reduced_instance)
                    norm_train = reduced_train
                    norm_test = reduced_test
                    dim_reduced_instances.append(dim_reduced_instance)


                # fit and predict
                train_y_frame = train_y_frame.to_numpy()
                train_y_frame = train_y_frame.reshape((train_y_frame.shape[0],))
                xgb_model.fit(norm_train, train_y_frame)
                # print('R2 score: ' + str(lrm_model.score(norm_train, train_y_frame)))
                xgb_pred = xgb_model.predict(norm_test)


                # compute rmsec
                rmsec_val = np.sqrt(mean_squared_error(test_y_frame, xgb_pred))
                train_rmsecs.append(rmsec_val)
                # print('Iter ' + str(i) + ' done')

            rmsec = round(np.mean(train_rmsecs), 4)
            rmsecs.append(rmsec)

            rmsecv = round(np.sqrt(mean_squared_error(df_v_y, xgb_model.predict(df_v_x))), 4)
            rmsecvs.append(rmsecv)

            xgb_models.append(xgb_model)
            param_set['model_id'] = m_id
            m_id += 1
            parameter_dictionaries.append(param_set)

            t_prime = dt.now()
            print('Training param set ' + str(m_id) + ' complete: ' + str(t_prime))
            
        
        model_opt = ModelOptimizer()
        opt_dict = model_opt.opt_params(rmsecs, rmsecvs, parameter_dictionaries)

        print('Train done')


        return (opt_dict, opt_dict['rmsec'], xgb_models, rmsecs, rmsecvs, parameter_dictionaries)
    
    def xgb_test(self, test_set, predictor_idx, target_idx, xgb_models, params, test_plot=False):
        
        test_set_x = test_set.iloc[:,predictor_idx[0]:predictor_idx[1]]
        test_set_y = test_set.iloc[:,target_idx]

        norm = Normalizer('mean-centering')
        norm_test = norm.normalize(test_set_x, [0,target_idx-1])

        rmsets = []

        for model in xgb_models:
            xgb_pred = model.predict(norm_test)
            rmset = np.sqrt(mean_squared_error(test_set_y, xgb_pred))
            rmsets.append(round(rmset, 4))

        if test_plot:
            self.test_plot(test_set_y, xgb_pred)

        return rmsets