import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

def split_data(dataset,test_size=0.2,seed=0):
    train_idx, test_idx, _, _ = train_test_split(range(len(dataset)),
                                                dataset.Y,
                                                stratify=dataset.Y,
                                                test_size=test_size,
                                                random_state=seed)

    return Subset(dataset,train_idx), Subset(dataset,test_idx)

class caseControlDataset(Dataset):
    def __init__(self,case,pheno_path,conn_path,dim=2,seed=0):
        """
        Parameters
        ----------
        case: str
        control: str
        pheno_path: str
        conn_path: str
        dim: int
            1: X is a vector, 2: X is a 2d array (needed for conv models).
        seed: int
        """
        assert ((dim == 1)|(dim == 2))
        self.dim = dim
        self.seed = seed
        self.name = case

        control = 'non_carriers'
        if case in ['SZ','ADHD','BIP','ASD']:
            control = 'CON_IPC'

        pheno = pd.read_csv(pheno_path,index_col=0)
        conn = pd.read_csv(conn_path,index_col=0)
        
        pis = pheno[pheno[case]==1]['PI'].unique()
        idx = pheno[(pheno['PI'].isin(pis))&((pheno[case]==1)|pheno[control]==1)].index
        self.X = conn.loc[idx].values
        self.Y = pheno.loc[idx][case].values.astype(int)

        del pheno
        del conn

    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self,idx):
        """
        Returns
        -------
        x: array of dim
        Y: dict str:vec
        """
        vec = self.X[idx,:]
        if self.dim == 1:
            return vec, {self.name:self.Y[idx]}
        else:
            np.random.seed(self.seed)
            #return vec[torch.randperm(2080)].reshape(40,52), self.Y[idx]
            return vec[np.random.permutation(2080)].reshape(40,52), {self.name:self.Y[idx]}

class ukbbSexDataset(Dataset):
    # TODO: random permutation seed?
    def __init__(self,pheno_path,conn_path,dim=2,seed=0):
        assert ((dim == 1)|(dim == 2))
        self.dim = dim
        self.seed = seed

        pheno = pd.read_csv(pheno_path,index_col=0)
        conn = pd.read_csv(conn_path,index_col=0)
        
        idx = pheno[pheno['PI']=='UKBB'].index
        self.X = conn.loc[idx].values
        self.Y = pheno.loc[idx]['SEX'].map({'Female':0,'Male':1}).values

        del pheno
        del conn

    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self,idx):
        vec = self.X[idx,:]
        if self.dim == 1:
            return vec, self.Y[idx]
        else:
            np.random.seed(self.seed)
            #return vec[torch.randperm(2080)].reshape(40,52), self.Y[idx]
            return vec[np.random.permutation(2080)].reshape(40,52), self.Y[idx]
