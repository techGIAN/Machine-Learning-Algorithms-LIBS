# imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

class DimensionReducer:

    n_comp = 0

    def __init__(self, n_comp):
        '''
            Initializes the Dimension Reducer with n_comp components. Method: PCA
            Params
            ------
            n_comp: number of components
        '''
        self.n_comp = n_comp
    
    def set_n_comp(self, n_comp):
        '''
            Sets the number of components
            Params
            ------
            n_comp: number of components
        '''
        self.n_comp = n_comp
    
    def dim_reduce_train(self, dataset, pred_idx, kernel):
        '''
            Reduces the dimension of the dataset into n_components.
            Params
            ------
            dataset: The training dataset
            pred_idx: The indices of the predictors
            Return
            ------
            A tuple: (pca, dataset_prime), where PCA is the learned PCA instance and dataset_prime is the dimensionally reduced training dataset
        '''
        df = dataset.copy(deep=True)
        df_predictors = df.iloc[:,pred_idx[0]:pred_idx[1]]
        df_before_pred = df.iloc[:,0:pred_idx[0]]
        df_after_pred = df.iloc[:,pred_idx[1]:]

        df_predictors = df_predictors.to_numpy()

        pca = KernelPCA(n_components=self.n_comp, kernel=kernel, remove_zero_eig=True)
        pca_instance = pca.fit(df_predictors)
        principal_comp = pca.transform(df_predictors)

        cols = ['x_' + str(i) for i in range(1,self.n_comp+1)]
        pdf = pd.DataFrame(principal_comp, columns=cols)
        new_df = pd.concat([df_before_pred, pdf, df_after_pred], axis=1)

        return (pca_instance, new_df)

    def dim_reduce_test(self, dataset, pred_idx, pca):
        '''
            Reduces the dimension of the dataset into n_components.
            Params
            ------
            dataset: The testing dataset
            pred_idx: The indices of the predictors
            pca: The PCA instance
            Return
            ------
            A dimensionally reduced testing set
        '''
        df = dataset.copy(deep=True)
        df_predictors = df.iloc[:,pred_idx[0]:pred_idx[1]]
        df_before_pred = df.iloc[:,0:pred_idx[0]]
        df_after_pred = df.iloc[:,pred_idx[1]:]

        df_predictors = df_predictors.to_numpy()
        principal_comp = pca.transform(df_predictors)

        cols = ['x_' + str(i) for i in range(1,self.n_comp+1)]
        pdf = pd.DataFrame(principal_comp, columns=cols)
        new_df = pd.concat([df_before_pred, pdf, df_after_pred], axis=1)

        return new_df

