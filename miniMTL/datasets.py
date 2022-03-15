import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class caseControlDataset(Dataset):
    def __init__(self,case,pheno_path,conn_path,dim=2,seed=0):
        """
        Create a dataset for a given case/control group.
        - Controls
            - idiopathic conditions: uses all 'CON_IPC' as controls
            - CNVs: uses 'non_carriers' from each PI
        - `case` becomes the dataset name
            - must match task names used to define HPSModel
        - Labels are returned as a dictionary from dataset name to array of int
        TODO: construct groups more carefully.

        Parameters
        ----------
        case: str
            Label of case to build dataset for (e.g. 'DEL22q11_2').
        pheno_path: str
            Path to phenotype .csv file.
        conn_path: str
            Path to directory containing connectomes (in square format).
        dim: int
            1: X is a vector, 2: X is a 2d array (needed for conv models).
        seed: int
            Seed to fix the random shuffle of the vector into 2d array.
        """
        assert ((dim == 1)|(dim == 2))
        self.dim = dim
        self.seed = seed
        self.name = case

        control = 'non_carriers'
        if case in ['SZ','ADHD','BIP','ASD']:
            control = 'CON_IPC'
        conn_path = os.path.join(conn_path,'connectome_{}_cambridge64.npy')

        pheno = pd.read_csv(pheno_path,index_col=0)
        
        pis = pheno[pheno[case]==1]['PI'].unique()
        idx = pheno[(pheno['PI'].isin(pis))&((pheno[case]==1)|pheno[control]==1)].index
        mask = np.tri(64,dtype=bool)

        self.X = np.array([np.load(conn_path.format(sub_id))[mask] for sub_id in idx])
        self.Y = pheno.loc[idx][case].values.astype(int)

        del pheno

    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self,idx):
        """
        Returns
        -------
        X: array
            Either a vector or randomly shuffled 2d array.
        Y: dict[str:array]
            Dictionary from dataset name to labels (array of int).
        """
        vec = self.X[idx,:]
        if self.dim == 1:
            return vec, {self.name:self.Y[idx]}
        else:
            np.random.seed(self.seed)
            return vec[np.random.permutation(2080)].reshape(40,52), {self.name:self.Y[idx]}

class ukbbSexDataset(Dataset):
    def __init__(self,pheno_path,conn_path,dim=2,seed=0):
        assert ((dim == 1)|(dim == 2))
        self.dim = dim
        self.seed = seed
        self.name = 'ukbb_sex'

        conn_path = os.path.join(conn_path,'connectome_{}_cambridge64.npy')
        pheno = pd.read_csv(pheno_path,index_col=0)

        idx = pheno[pheno['PI']=='UKBB'].index
        mask = np.tri(64,dtype=bool)

        self.X = np.array([np.load(conn_path.format(sub_id))[mask] for sub_id in idx])
        self.Y = pheno.loc[idx]['SEX'].map({'Female':0,'Male':1}).values

        del pheno

    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self,idx):
        vec = self.X[idx,:]
        if self.dim == 1:
            return vec, {self.name:self.Y[idx]}
        else:
            np.random.seed(self.seed)
            return vec[np.random.permutation(2080)].reshape(40,52), {self.name:self.Y[idx]}
