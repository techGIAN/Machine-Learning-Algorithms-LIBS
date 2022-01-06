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

from sklearn import linear_model as lm


import itertools
import warnings

class LRMModel:

    seed = 123
    regulatization = None
    
    def __init__(self, regularization=None, seed=123):
        '''
            Initializes the SVRModel Instance
        '''
        self.seed = seed
        self.regularization = regularization

    def lrm_train(self, df_tuple, df_val, predictor_idx, target_idx, params, dim_reduce=-1):

        warnings.filterwarnings('error')

        np.random.seed(self.seed)

        if self.regularization != 'linreg':
            parameter_set = [v for k, v in params.items()]
            param_names = list(params.keys())
            parameter_tuples = list(itertools.product(*parameter_set))

        else:
            parameter_tuples = range(1)
        

        predictor_tuple = tuple()
        target_tuple = tuple()
        for frame in df_tuple:
            predictor_frame = frame.iloc[:,predictor_idx[0]:predictor_idx[1]]
            predictor_tuple += (predictor_frame,)

            target_frame = frame.iloc[:,target_idx]
            target_tuple += (target_frame,)

        df_v_x = df_val.iloc[:,predictor_idx[0]:predictor_idx[1]]
        df_v_y = df_val.iloc[:,target_idx]
        
        nums = range(5)
        train_rmsecs = []
        rmsecs = []
        rmsecvs = []
        lrm_models = []

        best_model = None
        best_rmsec = np.inf

        num_iter = 0

        dim_reduced_instances = []

        parameter_dictionaries = []

        m_id = 0

        for param_tup in parameter_tuples:

            try:
                rmsec = []

                if self.regularization != 'linreg':
                    params_dict = dict(zip(param_names, param_tup))

                if self.regularization == 'lasso':
                    lrm_model = lm.Lasso()
                elif self.regularization == 'ridge':
                    params_dict['solver'] = 'svd'
                    lrm_model = lm.Ridge()
                elif self.regularization == 'enets':
                    lrm_model = lm.ElasticNet()
                else:
                    lrm_model = lm.LinearRegression(fit_intercept=True)
                
                if self.regularization != 'linreg':
                    lrm_model.set_params(**params_dict)
                

                for i in nums:

                    norm = Normalizer('mean-centering')

                    test_x_frame = predictor_tuple[i].reset_index().drop(['index'], axis=1)
                    test_y_frame = target_tuple[i].reset_index().drop(['index'], axis=1)

                    train_nums = [i for n in nums if i != n]
                    trainers_x = [predictor_tuple[tn] for tn in train_nums]
                    trainers_y = [target_tuple[tn] for tn in train_nums]
                    train_x_frame = trainers_x[0].append([trainers_x[1], trainers_x[2], trainers_x[3]], ignore_index=True)
                    train_y_frame = trainers_y[0].append([trainers_y[1], trainers_y[2], trainers_y[3]], ignore_index=True)

                    norm_train = norm.normalize(train_x_frame, [0,target_idx-1])
                    norm_test = norm.normalize(test_x_frame, [0,target_idx-1])


                    train_y_frame = train_y_frame.to_numpy()
                    train_y_frame = train_y_frame.reshape((train_y_frame.shape[0],))
                    lrm_model.fit(norm_train, train_y_frame)
                    lrm_pred = lrm_model.predict(norm_test)

                    rmsec_val = np.sqrt(mean_squared_error(test_y_frame, lrm_pred))
                    train_rmsecs.append(rmsec_val)

                rmsec = round(np.mean(train_rmsecs), 4)
                rmsecs.append(rmsec)

                rmsecv = round(np.sqrt(mean_squared_error(df_v_y, lrm_model.predict(df_v_x))), 4)
                rmsecvs.append(rmsecv)

            
                lrm_models.append(lrm_model)
                
                if self.regularization != 'linreg':
                    params_dict['model_id'] = m_id
                    m_id += 1
                    parameter_dictionaries.append(params_dict)
                    print('Model ' + str(m_id) + ' done')
                
                num_iter += 1
                t_prime = dt.now()
                print('Training param set ' + str(num_iter) + ' complete: ' + str(t_prime))
                
            except:
                continue
        
        model_opt = ModelOptimizer()

        if self.regularization != 'linreg':
            opt_dict = model_opt.opt_params(rmsecs, rmsecvs, parameter_dictionaries)
        else:
            opt_dict = {
                    'index' : 0,
                    'rmsec': rmsecs[0],
                    'rmsecv': rmsecvs[0],
                    'params': {'model_id':0}
                    }
            parameter_dictionaries = [opt_dict]

        print('Train done')

        return (opt_dict, opt_dict['rmsec'], lrm_models, rmsecs, rmsecvs, parameter_dictionaries)
    
    def lrm_test(self, test_set, predictor_idx, target_idx, lrm_models, params, test_plot=False):
        
        test_set_x = test_set.iloc[:,predictor_idx[0]:predictor_idx[1]]
        test_set_y = test_set.iloc[:,target_idx]

        norm = Normalizer('mean-centering')
        norm_test = norm.normalize(test_set_x, [0,target_idx-1])

        rmsets = []

        for model in lrm_models:
            lrm_pred = model.predict(norm_test)
            rmset = np.sqrt(mean_squared_error(test_set_y, lrm_pred))
            rmsets.append(round(rmset, 4))

        if test_plot:
            self.test_plot(test_set_y, lrm_pred)

        return rmsets