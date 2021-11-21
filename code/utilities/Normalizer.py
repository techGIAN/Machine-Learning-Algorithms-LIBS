# imports
import pandas as pd
import numpy as np

class Normalizer:

    method = None
    methods = ['z-score', 'mean-centering', 'min-max-normalization', 'feature-clipping', 'log-scaling']         # there's more; not all implemented yet
    x_min = 0       # only for min-max norm
    x_max = 1       # only for min-max norm

    def __init__(self, method='z-score', x_min=0, x_max=1):
        '''
            Initializes the Normalizer
            Params
            ------
            method: the method for normalizing (optional; 'z-score' by default)
            x_min: minimum for min-max norm (0 by default)
            x_max: maximum for min-max norm (1 by default)
        '''
        self.set_method(method)
        self.x_min = x_min
        self.x_max = x_max
    
    def set_method(self, method):
        '''
            Sets the method of normalization.
            Params
            ------
            method: the method
        '''
        if method in self.methods:
            self.method = method
        else:
            print('The provided method is not available. Try again')
    
    def set_min_max_parameters(self, x_min, x_max):
        '''
            Sets the min-max parameters
            Params
            ------
            x_min: minimum for min-max norm (0 by default)
            x_max: maximum for min-max norm (1 by default)
        '''
        if self.method == 'min-max-normalization':
            self.x_min = x_min
            self.x_max = x_max
        else:
            print('Not applicatble.')
    
    def normalize(self, dataset, col_idx=None):
        '''
            Normalizes the provided dataset.
            Params
            ------
            dataset: The dataset to be normalized
            col_idx: The specific column indices to normalize in the dataset (list inclusive, exclusive)
        '''
        if self.method == 'z-score':
            return self.__z_score(dataset, col_idx)
        elif self.method == 'mean-centering':
            return self.__mean_centering(dataset, col_idx)
        elif self.method == 'min-max-normalization':
            return self.__min_max(dataset, col_idx)
        else:
            return dataset      # default; nothing to return
    
    def __z_score(self, dataset, col_idx):
        '''
            Normalizes using z-score
            Params
            ------
            dataset: The dataset to be normalized using z-score
            col_idx: The specific column indices to normalize in the dataset (list inclusive, exclusive)
        '''
        df = dataset.copy(deep=True)
        means = df.mean(axis=0, skipna=True)
        stds = df.std(axis=0, skipna=True)
        
        columns = df.columns.tolist()[col_idx[0] : col_idx[1]] if col_idx != None else df.select_dtypes(include=np.number).columns.tolist()

        for col in columns:
            df[col] = (df[col] - means[col]) / stds[col]
        
        return df
    
    def __mean_centering(self, dataset, col_idx):
        '''
            Normalizes using mean-centering
            Params
            ------
            dataset: The dataset to be normalized using mean-centering
            col_idx: The specific column indices to normalize in the dataset (list inclusive, exclusive)
        '''
        df = dataset.copy(deep=True)
        means = df.mean(axis=0, skipna=True)

        columns = df.columns.tolist()[col_idx[0] : col_idx[1]] if col_idx != None else df.select_dtypes(include=np.number).columns.tolist()

        for col in columns:
            df[col] = df[col] - means[col]
        
        return df
    
    def __min_max(self, dataset, col_idx):
        '''
            Normalizes using min-max normalization
            Params
            ------
            dataset: The dataset to be normalized using min-max normalization
            col_idx: The specific column indices to normalize in the dataset (list inclusive, exclusive)
        '''
        df = dataset.copy(deep=True)
        min_col = df.min(axis=0, skipna=True)
        max_col = df.max(axis=0, skipna=True)
        new_range = self.x_max - self.x_min

        columns = df.columns.tolist()[col_idx[0] : col_idx[1]] if col_idx != None else df.select_dtypes(include=np.number).columns.tolist()

        for col in columns:
            df[col] = (df[col] - min_col[col]) / (max_col[col] - min_col[col]) * new_range + self.x_min
        
        return df
