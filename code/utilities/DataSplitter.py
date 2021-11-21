# imports
import pandas as pd
import numpy as np
import random as rnd

class DataSplitter:

    k_fold = 0
    val_prop = 0
    test_prop = 0
    ens_test_prop = 0
    seed = 123

    def __init__(self, k_fold=5, val_prop=0.1, test_prop=0.1, ens_test_prop=0.1, seed=123):
        '''
            Initializes DataSplitter.
            Params
            ------
            k_fold: number of disjoint groups in the training set (int)
            val_prop: proportion of the dataset that is validation set (float; between 0 and 1)
            test_prop: proportion of the dataset that is testing set (float; between 0 and 1)
            ens_test_prop: proportion of the dataset that is testing set for ensemble (float; between 0 and 1)
        '''
        self.k_fold = k_fold
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.ens_test_prop = ens_test_prop
        self.seed = seed

    def split_data(self, dataset, k_fold=5, val_prop=0.1, test_prop=0.1, ens_test_prop=0.1):
        '''
            Splits the given data according to the current DataSplitter() parameter. The parameters can also be changed here.
            Params
            ------
            dataset: the dataset
            k_fold: number of disjoint groups in the training set (int)
            val_prop: proportion of the dataset that is validation set (float; between 0 and 1)
            test_prop: proportion of the dataset that is testing set (float; between 0 and 1)
            ens_test_prop: proportion of the dataset that is testing set for ensemble (float; between 0 and 1)
            Return
            ------
            data_frames: A (k+3)-tuple, where the first k components are the k-folds that comprise the training set; T
            The (k+1)th component is the validation set; the (k+2)th component is the testing set.
            The (k+3)th is the testing set for ensemble. Each component is a pandas dataframe
        '''


        rnd.seed(self.seed)
        
        unique_samples = list(dataset.iloc[:,0].unique())
        n = len(unique_samples)
        ens_testing_samples = rnd.sample(unique_samples, int(ens_test_prop*float(n)))
        train_val_test_samples = [x for x in unique_samples if x not in ens_testing_samples]
        testing_samples = rnd.sample(train_val_test_samples, int(test_prop*float(n)))
        train_val_samples = [x for x in unique_samples if x not in testing_samples]
        validation_samples = rnd.sample(train_val_samples, int(val_prop*float(n)))
        training_samples = [x for x in train_val_samples if x not in validation_samples]

        array_training_samples = np.array_split(training_samples, k_fold)
        sample_group_mapper = dict()

        i = 1
        for array in array_training_samples:
            for sample in array:
                sample_group_mapper[sample] = 'Fold' + str(i)
            i += 1
        
        for sample in validation_samples:
            sample_group_mapper[sample] = 'Val'
        
        for sample in testing_samples:
            sample_group_mapper[sample] = 'Test'

        for sample in ens_testing_samples:
            sample_group_mapper[sample] = 'Ens_Test'

        df = dataset
        df['fold'] = 'None'
        for idx, row in df.iterrows():
            sample_name = row['sample']
            df.loc[df['sample'] == sample_name, 'fold'] = sample_group_mapper[sample_name]
        
        data_frames = tuple()

        for i in range(1,k_fold+1):
            frame = df.loc[df['fold'] == 'Fold' + str(i)]
            data_frames += (frame,)
        
        val_frame = df.loc[df['fold'] == 'Val']
        test_frame = df.loc[df['fold'] == 'Test']
        ens_test_frame = df.loc[df['fold'] == 'Ens_Test']

        data_frames += (val_frame, test_frame, ens_test_frame)
        
        return data_frames
