# imports
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt

# from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import regularizers

import tensorflow as tf

from utilities.Normalizer import Normalizer
from utilities.ModelOptimizer import ModelOptimizer
# from DimensionReducer import DimensionReducer
# from sklearn.decomposition import KernelPCA

# import xgboost as xgb


import itertools
import warnings

class NNModel:

    seed = 123
    
    def __init__(self, seed=123, model='ann'):
        '''
            Initializes the SVRModel Instance
        '''
        self.seed = seed if model == 'cnn' else 1001

    def nn_param_combos(self, param_dict, N=np.inf):
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
        rnd.seed(self.seed)
        # check for the validity of N
        if N < 0 or (int(N) != N and N > 1) or N == np.inf:
            print(str(N) + ' is not valid')
            exit()

        # generate a Cartesian product to generate all combinations
        parameter_set = [v for k, v in param_dict.items()]
        param_names = list(param_dict.keys())
        parameter_tuples = list(itertools.product(*parameter_set))

        param_combos = [] # keep all possible combinations to return

        for param_tup in parameter_tuples.copy():
            params_dict = dict(zip(param_names, param_tup))
            
            if params_dict['optimizer'] != 'adam':
                params_dict.pop('beta_1')
                params_dict.pop('beta_2')
                params_dict.pop('amsgrad')
            
            if params_dict['optimizer'] == 'sgd':
                params_dict.pop('epsilon')
            else:
                params_dict.pop('nesterov')
                params_dict.pop('momentum')

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


    def nn_train(self, df_tuple, df_val, predictor_idx, target_idx, params, mdl, dim_reduce=-1):

        N = 5 if mdl == 'ann' else 1
        parameter_combos = self.nn_param_combos(params, N=N)
        print(parameter_combos)
        
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

        df_v_y = df_v_y.to_numpy()
        df_v_x = df_v_x.to_numpy()
        
        # 5 folds
        nums = range(5)
        train_rmsecs = []
        rmsecs = []
        rmsecvs = []
        nn_models = []

        

        num_iter = 0

        parameter_dictionaries = []
        # for kern in params['kernel']:
        m_id = 0

        for param_set in parameter_combos:

            # dict_string = '{'
            # for k, v in param_set.items():
            #     dict_string += k + ': ' + str(v) + ', '
            # dict_string += '}'
            # dict_string = dict_string.replace(', }', '}')
            # print(dict_string)

            rmsec = []

            temp_param_set = param_set.copy()
            temp_param_set.pop('model_id')

            act_func = {'relu': tf.keras.layers.ReLU(),
                        'softmax': tf.keras.layers.Softmax(),
                        'leaky_relu': tf.keras.layers.LeakyReLU()}

            # ann_model = Sequential()
            # in_dim = 6144
            # if temp_param_set['with_conv']:
            #     ann_model.add(Conv1D(filters=temp_param_set['filters'], kernel_size=2, padding='valid',
            #                         activation=act_func[temp_param_set['activation']], input_shape=(in_dim,1)))
            #     ann_model.add(MaxPooling1D(pool_size=2))
            #     ann_model.add(Flatten())
            # for n_hidden_units in temp_param_set['hidden_units']:
            #     if not temp_param_set['with_conv']:
            #         ann_model.add(Dense(n_hidden_units, input_dim=in_dim, activation=act_func[temp_param_set['activation']]))
            #     else:
            #         ann_model.add(Dense(n_hidden_units, activation=act_func[temp_param_set['activation']]))
            #     in_dim = n_hidden_units
            #     ann_model.add(Dropout(temp_param_set['dropout']))
            # ann_model.add(Dense(1, activation=act_func[temp_param_set['activation']]))

            # optimizer = None
            # if temp_param_set['optimizer'] == 'rmsprop':
            #     optimizer = tf.keras.optimizers.RMSprop(learning_rate=temp_param_set['learning_rate'],
            #                                             epsilon=temp_param_set['epsilon'])
            # elif temp_param_set['optimizer'] == 'adam':
            #     optimizer = tf.keras.optimizers.Adam(learning_rate=temp_param_set['learning_rate'],
            #                                             epsilon=temp_param_set['epsilon'],
            #                                             beta_1=temp_param_set['beta_1'],
            #                                             beta_2=temp_param_set['beta_2'],
            #                                             amsgrad=temp_param_set['amsgrad'])
            # else:
            #     optimizer = tf.keras.optimizers.SGD(learning_rate=temp_param_set['learning_rate'],
            #                                             momentum=temp_param_set['momentum'],
            #                                             nesterov=temp_param_set['nesterov'])

            # ann_model.compile(loss='mean_squared_error', optimizer=optimizer)
            

            for i in nums:

                norm = Normalizer('mean-centering')
                # dr = DimensionReducer(n_comp=dim_reduce)

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

                nn_model = Sequential()
                in_dim = 6144
                if temp_param_set['with_conv']:
                    nn_model.add(Conv1D(filters=temp_param_set['filters'], kernel_size=2, padding='valid',
                                        activation=act_func[temp_param_set['activation']], input_shape=(norm_train.shape[1],1)))
                    nn_model.add(MaxPooling1D(pool_size=2))
                    nn_model.add(Flatten())
                    # in_dim = norm_train.shape[1]
                for n_hidden_units in temp_param_set['hidden_units']:
                    if not temp_param_set['with_conv']:
                        nn_model.add(Dense(n_hidden_units, input_dim=in_dim, activation=act_func[temp_param_set['activation']]))
                    else:
                        nn_model.add(Dense(n_hidden_units, activation=act_func[temp_param_set['activation']]))
                    in_dim = n_hidden_units
                    nn_model.add(Dropout(temp_param_set['dropout']))
                nn_model.add(Dense(1, activation=act_func[temp_param_set['activation']]))

                optimizer = None
                if temp_param_set['optimizer'] == 'rmsprop':
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=temp_param_set['learning_rate'],
                                                            epsilon=temp_param_set['epsilon'])
                elif temp_param_set['optimizer'] == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=temp_param_set['learning_rate'],
                                                            epsilon=temp_param_set['epsilon'],
                                                            beta_1=temp_param_set['beta_1'],
                                                            beta_2=temp_param_set['beta_2'],
                                                            amsgrad=temp_param_set['amsgrad'])
                else:
                    optimizer = tf.keras.optimizers.SGD(learning_rate=temp_param_set['learning_rate'],
                                                            momentum=temp_param_set['momentum'],
                                                            nesterov=temp_param_set['nesterov'])

                nn_model.compile(loss='mean_squared_error', optimizer=optimizer)


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
                test_y_frame = test_y_frame.to_numpy()
                test_y_frame = test_y_frame.reshape((test_y_frame.shape[0],))
                if temp_param_set['with_conv']:
                    new_dim_0 = norm_train.shape[0]
                    new_dim_1 = norm_train.shape[1]
                    norm_train = norm_train.to_numpy().reshape(new_dim_0, new_dim_1, 1)
                    nn_model.fit(norm_train, train_y_frame, batch_size=16)
                else:
                    nn_model.fit(norm_train, train_y_frame, batch_size=128)

                if temp_param_set['with_conv']:
                    new_dim_0 = norm_test.shape[0]
                    new_dim_1 = norm_test.shape[1]
                    norm_test = norm_test.to_numpy().reshape(new_dim_0, new_dim_1, 1)
                    ann_score = nn_model.evaluate(norm_test, test_y_frame, batch_size=16)
                else:
                    ann_score = nn_model.evaluate(norm_test, test_y_frame, batch_size=128)


                # compute rmsec
                rmsec_val = np.sqrt(ann_score)
                train_rmsecs.append(rmsec_val)
                # print('Iter ' + str(i) + ' done')

            rmsec = round(np.mean(train_rmsecs), 4)
            rmsecs.append(rmsec)

            df_v_y = df_v_y.reshape((df_v_y.shape[0],))
            if temp_param_set['with_conv']:
                new_dim_0 = df_v_x.shape[0]
                new_dim_1 = df_v_x.shape[1]
                df_v_x = df_v_x.reshape(new_dim_0, new_dim_1, 1)
                ann_score = nn_model.evaluate(df_v_x, df_v_y, batch_size=16)
            else:
                ann_score = nn_model.evaluate(df_v_x, df_v_y, batch_size=128)

            rmsecv = round(np.sqrt(ann_score), 4)
            rmsecvs.append(rmsecv)

            nn_models.append(nn_model)
            param_set['model_id'] = m_id
            m_id += 1
            parameter_dictionaries.append(param_set)

            t_prime = dt.now()
            print('Training param set ' + str(m_id) + ' complete: ' + str(t_prime))
            
        
        model_opt = ModelOptimizer()
        opt_dict = model_opt.opt_params(rmsecs, rmsecvs, parameter_dictionaries)


        print('Train done')

        return (opt_dict, opt_dict['rmsec'], nn_models, rmsecs, rmsecvs, parameter_dictionaries)
    
    def nn_test(self, test_set, predictor_idx, target_idx, ann_models, params, test_plot=False):
 
        test_set_x = test_set.iloc[:,predictor_idx[0]:predictor_idx[1]]
        test_set_y = test_set.iloc[:,target_idx]

        norm = Normalizer('mean-centering')
        norm_test = norm.normalize(test_set_x, [0,target_idx-1])

        rmsets = []

        i = 0
        for model in ann_models:

            if params[i]['with_conv']:
                new_dim_0 = norm_test.shape[0]
                new_dim_1 = norm_test.shape[1]
                norm_test = norm_test.to_numpy().reshape(new_dim_0, new_dim_1, 1)

            y_shape = test_set_y.shape[0]
            test_set_y = np.array(list(test_set_y))
            test_set_y = test_set_y.reshape((y_shape,))
            if not params[i]['with_conv']:
                ann_score = model.evaluate(norm_test, test_set_y, batch_size=128)
            else:
                ann_score = model.evaluate(norm_test, test_set_y, batch_size=16)

            rmset = np.sqrt(ann_score)
            rmsets.append(round(rmset, 4))

            i += 1

        return rmsets