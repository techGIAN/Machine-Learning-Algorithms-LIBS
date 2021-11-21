# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from utilities.Normalizer import Normalizer
from utilities.ModelOptimizer import ModelOptimizer
from sklearn.decomposition import KernelPCA

import itertools
import warnings

class SVRModel:

    seed = 123
    
    def __init__(self, seed=123):
        '''
            Initializes the SVRModel Instance
        '''
        self.seed = seed

    def svr_train(self, df_tuple, df_val, predictor_idx, target_idx, params, dim_reduce=-1):

        warnings.filterwarnings('error')

        np.random.seed(self.seed)

        parameter_set = [v for k, v in params.items()]
        param_names = list(params.keys())
        parameter_tuples = list(itertools.product(*parameter_set))
        parameter_tuples = [tup for tup in parameter_tuples if tup[0] == 'poly' or
                                                                (tup[0] != 'poly' and tup[1] == 3)]
        parameter_tuples = [tup for tup in parameter_tuples if tup[0] != 'rbf' or
                                                                (tup[0] == 'poly' and tup[3] == 0.0)]
        
        # take a subset
        parameter_tuples = rnd.sample(parameter_tuples, k=int(0.01*len(parameter_tuples)))


        # remove all unnecessary columns
        predictor_tuple = tuple()
        target_tuple = tuple()
        for frame in df_tuple:
            predictor_frame = frame.iloc[:,predictor_idx[0]:predictor_idx[1]]
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
        svr_models = []

        num_iter = 0
        # max_runs = len(params['kernel'])

        # while num_iter < max_runs:

        parameter_dictionaries = []
        # for kern in params['kernel']:
        m_id = 0
        for param_tup in parameter_tuples:

            try:
                rmsec = []

                params_dict = dict(zip(param_names, param_tup))
                svr_model = SVR()
                svr_model.set_params(**params_dict)
                

                for i in nums:

                    norm = Normalizer('mean-centering')

                    test_x_frame = predictor_tuple[i].reset_index().drop(['index'], axis=1)
                    test_y_frame = target_tuple[i].reset_index().drop(['index'], axis=1)

                    train_nums = [i for n in nums if i != n]
                    trainers_x = [predictor_tuple[tn] for tn in train_nums]
                    trainers_y = [target_tuple[tn] for tn in train_nums]
                    train_x_frame = trainers_x[0].append([trainers_x[1], trainers_x[2], trainers_x[3]], ignore_index=True)
                    train_y_frame = trainers_y[0].append([trainers_y[1], trainers_y[2], trainers_y[3]], ignore_index=True)

                    # print('Training and testing selected done')

                    # normalize
                    norm_train = norm.normalize(train_x_frame, [0,target_idx-1])
                    norm_test = norm.normalize(test_x_frame, [0,target_idx-1])

                    # fit and predict
                    train_y_frame = train_y_frame.to_numpy()
                    train_y_frame = train_y_frame.reshape((train_y_frame.shape[0],))
                    svr_model.fit(norm_train, train_y_frame)
                    svr_pred = svr_model.predict(norm_test)

                    # print('Fitting and prediction done')

                    # compute rmsec
                    rmsec_val = np.sqrt(mean_squared_error(test_y_frame, svr_pred))
                    train_rmsecs.append(rmsec_val)
                    # print('Iter ' + str(i) + ' done')

                rmsec = round(np.mean(train_rmsecs), 4)
                rmsecs.append(rmsec)

                rmsecv = round(np.sqrt(mean_squared_error(df_v_y, svr_model.predict(df_v_x))), 4)
                rmsecvs.append(rmsecv)
            
                svr_models.append(svr_model)
                params_dict['model_id'] = m_id
                m_id += 1
                parameter_dictionaries.append(params_dict)
                
                num_iter += 1
                t_prime = dt.now()
                # print(str(params_dict))
                print('Training param set ' + str(num_iter) + ' complete: ' + str(t_prime))
        
            except:
                # print('Error found')
                continue
        
        model_opt = ModelOptimizer()
        opt_dict = model_opt.opt_params(rmsecs, rmsecvs, parameter_dictionaries)

        print('Train done')

        # return (min_rmsec, svr_models, rmsecs, opt_params, dim_reduced_instances, parameter_dictionaries)
        return (opt_dict, opt_dict['rmsec'], svr_models, rmsecs, rmsecvs, parameter_dictionaries)

    def svr_test(self, test_set, predictor_idx, target_idx, svr_models, params, test_plot=False):

        test_set_x = test_set.iloc[:,predictor_idx[0]:predictor_idx[1]]
        test_set_y = test_set.iloc[:,target_idx]

        norm = Normalizer('mean-centering')
        norm_test = norm.normalize(test_set_x, [0,target_idx-1])

        rmsets = []

        for model in svr_models:
            svr_pred = model.predict(norm_test)
            rmset = np.sqrt(mean_squared_error(test_set_y, svr_pred))
            rmsets.append(round(rmset, 4))

        if test_plot:
            self.test_plot(test_set_y, svr_pred)

        return rmsets