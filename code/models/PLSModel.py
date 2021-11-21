# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random as rnd

from os import listdir
from os.path import isfile, join

from utilities.Normalizer import Normalizer
from utilities.ModelOptimizer import ModelOptimizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

class PLSModel:

    seed = 123

    def __init__(self, seed=123):
        '''
            Initializes a PLSModel instance
        '''
        self.seed = seed
    
    def pls_train(self, df_tuple, df_val, predictor_idx, target_idx, component_bound=np.arange(2,51)):

        np.random.seed(self.seed)
        
        norm = Normalizer('mean-centering')

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
        n_comp = np.arange(2, min(max(component_bound), predictor_frame.shape[1])+1)
        
        # 5 folds
        nums = range(5)
        train_rmsecs = []
        rmsecs = []
        rmsecvs = []
        pls_models = []

        best_model = None
        best_rmsec = np.inf

        num_iter = 0
        max_runs = len(n_comp)

        while num_iter < max_runs:

            rmsec = []
            pls_model = PLSRegression(n_components=n_comp[num_iter], scale=False,
                                            max_iter=500, tol=1e-06)
            for i in nums:

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

                # fit and predict
                pls_model.fit(norm_train, train_y_frame)
                pls_pred = pls_model.predict(norm_test)

                # compute rmsec
                rmsec_val = np.sqrt(mean_squared_error(test_y_frame, pls_pred))
                train_rmsecs.append(rmsec_val)

            rmsec = round(np.mean(train_rmsecs), 4)
            rmsecs.append(rmsec)

            rmsecv = round(np.sqrt(mean_squared_error(df_v_y, pls_model.predict(df_v_x))), 4)
            rmsecvs.append(rmsecv)

            pls_models.append(pls_model)
            
            num_iter += 1
            print('Component ' + str(num_iter+1) + ' complete.')


        param_dictionaries = [{'n_components': x} for x in range(n_comp[0], n_comp[-1]+1)]

        model_opt = ModelOptimizer()
        opt_dict = model_opt.opt_params(rmsecs, rmsecvs, param_dictionaries)
        

        return (opt_dict, opt_dict['rmsec'], pls_models, rmsecs, rmsecvs, opt_dict['params']['n_components'])
    
    def pls_test(self, test_set, predictor_idx, target_idx, n_comps, pls_models=None, test_plot=False):
        test_set_x = test_set.iloc[:,predictor_idx[0]:predictor_idx[1]]
        test_set_y = test_set.iloc[:,target_idx]

        norm = Normalizer('mean-centering')
        norm_test = norm.normalize(test_set_x, [0,target_idx-1])

        rmsets = []

        if pls_models == None:
            pls_model = PLSRegression(n_components=n_comps, scale=False,
                                            max_iter=500, tol=1e-06)
            pls_models = [pls_model]

        for model in pls_models:
            pls_pred = model.predict(norm_test)
            rmset = np.sqrt(mean_squared_error(test_set_y, pls_pred))
            rmsets.append(round(rmset, 4))

        if test_plot:
            self.test_plot(test_set_y, pls_pred)

        return rmsets

    def test_plot(self, actual_y, predicted_y):
        '''
            Plot the Actual vs Predicted Y (only for the validation or testing plot)
        '''
        actual_values = [x[0] for x in actual_y.values.tolist()]
        predicted_values = [x[0] for x in predicted_y]
        min_actual_y = min(actual_values)
        max_actual_y = max(actual_values) + 1
        
        x_label = 'Observed SiO2'
        y_label = 'Predicted SiO2'
        title = 'Validation for PLS SiO2'

        # Plot y = x
        x_range = np.arange(min_actual_y, max_actual_y)
        y_range = x_range
        plt.plot(x_range, y_range, color='k', linestyle='-')

        # scatter plot of the points
        plt.scatter(actual_y, predicted_y, s=5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        # if save_plot:
        f_name = self.get_plot_filename()
        plt.savefig(f_name)

        # plt.show()
    
    def get_plot_filename(self):
        '''
            Gets the filename of the plot
        '''
        out_dir = './outputs'
        files = [f for f in listdir(out_dir) if isfile(join(out_dir, f))]
        files = [f[:-4] for f in files if f[0:3] == 'out']
        nums = [int(f[3:]) for f in files]
        max_num = max(nums)+1
        max_num = str(max_num) if max_num >= 10 else '0' + str(max_num)
        f_out = './outputs/out' + max_num + '.png'

        return f_out