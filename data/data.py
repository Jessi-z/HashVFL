import os.path
import random
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Sampler

from data.dataset import Textdataset, Tabdataset

def loadDataset(dataset_name, num_party, num_features, max_length=80):
    database = '/home/qpy/datasets' if os.path.exists(
        '/home/qpy/datasets') else 'C:\\Users\\Qiupys\\PycharmProjects\\datasets'
    tran_datasets, test_datasets = [], []
    
    if dataset_name == 'criteo':
        columns = ['label', *(f'I{i}' for i in range(1, 14)), *(f'C{i}' for i in range(1, 27))]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        df = pd.read_csv(os.path.join(database, 'criteo/dac_sample.txt'), sep='\t')
        # , names=columns).fillna(method='bfill', axis=0)
        df[dense_features] = df[dense_features].fillna(0)
        df[sparse_features] = df[sparse_features].fillna('-1')
        # df.dropna(how='any', axis=0, inplace=True)

        # Preprocessing Integer Features
        integer_cols = dense_features
        df[integer_cols] = preprocessing.MinMaxScaler().fit_transform(df[integer_cols])

        # Preprocess Categorical Features
        remove_list = ['C3', 'C4', 'C7', 'C10', 'C11', 'C12', 'C13', 'C15', 'C16', 'C18', 'C19', 'C21', 'C24', 'C26']
        df = df.drop(remove_list, axis=1)
        cat_cols = [c for c in sparse_features and c not in remove_list]
        df[cat_cols] = df[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)
        # for col in cat_cols:
        #     preprocessing.OneHotEncoder(sparse=False).fit_transform(df[[col]])

        X, y = df.drop('label', axis=1), df['label'].astype(int)
        logging.info(y.value_counts())
        # # Preprocessing Label Imbalance
        # from imblearn.under_sampling import RandomUnderSampler
        # rus = RandomUnderSampler()
        # X, y = rus.fit_resample(X, y)
        # logging.info(y.value_counts())
        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)
        logging.info(y.value_counts())

        # Generate Dataset
        X, y = np.array(X), np.array(y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        train_datasets, test_datasets = [], []
        start_index = 0
        for i in range(num_party):
            train_datasets.append(
                Tabdataset(X_train, label=y_train, start=start_index, end=start_index + num_features[i]))
            test_datasets.append(Tabdataset(X_test, label=y_test, start=start_index, end=start_index + num_features[i]))
            start_index += num_features[i]
        return train_datasets, test_datasets
    else:
        logging.info('Dataset does not exist!!!')
        exit()
    return tran_datasets, test_datasets


def dataLoader(datasets, batch_size):
    loaders = []
    order = list(range(len(datasets[0])))
    random.shuffle(order)

    class MySampler(Sampler):
        r"""Samples elements according to the previously generated order.
        """

        def __init__(self, data_source, order):
            super().__init__(data_source)
            self.data_source = data_source
            self.order = order

        def __iter__(self):
            return iter(self.order)

        def __len__(self):
            return len(self.data_source)

    for dataset in datasets:
        loaders.append(
            DataLoader(dataset, batch_size, shuffle=False, sampler=MySampler(dataset, order), drop_last=True))
    return loaders
