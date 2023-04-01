'''
Author: Manami Kanemura
'''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, sampler, Dataset

class Custom_Dataset(Dataset):
    '''
    class: custom_dataset
    does: create a custom dataset
    parameters: df.data, df.label
    '''
    
    def __init__(self, label, data):
        self.label = label
        self.data = data
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        label = self.label[idx]
        data = self.data.iloc[idx, :].to_numpy()
        sample = {"data": data, "target": label}
        
        return sample
    
def load_dataloader(source=['statslib', 'gbreb']):
    '''
    function: load_dataloader
    does: (1) retreive csv data, (2)split the data into data and label, (3)create a custom dataset
          (4) create a dataloader
    parameter: source of dataset (either statslib or gbreb)
    return: dataloader 
    '''
    
    ## import the dataset
    if source == 'statslib':
        f_path = 'data/selected_statslib.csv'
    elif source == 'gbreb':
        f_path = 'data/selected_GBREB.csv'
    else:
        print('specify the source of dataset')
        
    df = pd.read_csv(f_path)
    df.drop(columns='Unnamed: 0', axis=1, inplace=True)
    
    ## prepare data and labels
    target = 'MEDV'
    X = df.drop(target, axis=1)
    y = df[target]
    
    if source == 'statslib':
        ## Create a custom dataset
        print('creating a custom dataset...')
        X.reset_index(inplace=True, drop=True)
        CD = Custom_Dataset(y, X)
        print(next(iter(CD)))
        
        ## create a dataloader
        print('creating a dataloader...')
        DL = DataLoader(CD, batch_size=10, shuffle=True)
        print('>>>> Check the DL <<<<')
        for (idx, batch) in enumerate(DL):
            print(idx, batch['data'])
            print(batch['target'])
            break
            
        return DL

    elif source == 'gbreb':
        '''
        train data = 50% of total data
        val data = 20% of total data
        test data = 30% of total data
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        
        ## Create a custom dataset
        print('creating a custom dataset...')
        ## reset index
        X_train.reset_index(inplace=True, drop=True)
        X_val.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        y_val.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)
        
        gbreb_CD_train = Custom_Dataset(y_train, X_train)
        gbreb_CD_val = Custom_Dataset(y_val, X_val)
        gbreb_CD_test = Custom_Dataset(y_test, X_test)
        
        print(f'train data = {next(iter(gbreb_CD_train))}')
        print(f'validation data = {next(iter(gbreb_CD_val))}')
        print(f'test data = {next(iter(gbreb_CD_test))}')
    
        ## create a dataloader
        print('creating a dataloader...')
        GBREB_DL_train = DataLoader(gbreb_CD_train, batch_size=8, shuffle=True)
        GBREB_DL_val = DataLoader(gbreb_CD_val, batch_size=5, shuffle=True)
        GBREB_DL_test = DataLoader(gbreb_CD_test, batch_size=10, shuffle=True)
        
        ## check dataloader
        print('--- train data ---')
        for (idx, batch) in enumerate(GBREB_DL_train):
            print(idx, batch['data'])
            print(batch['target'])
            break
        print('--- validation data ---')
        for (idx, batch) in enumerate(GBREB_DL_val):
            print(idx, batch['data'])
            print(batch['target'])
            break
        print('--- test data ---')
        for (idx, batch) in enumerate(GBREB_DL_test):
            print(idx, batch['data'])
            print(batch['target'])
            break
            
        return GBREB_DL_train, GBREB_DL_val, GBREB_DL_test