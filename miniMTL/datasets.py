import pandas as pd
import numpy as np
import random
import os

import torch
from torch.utils.data import Dataset

def vec_to_connectome(a,dim=64):
    """
    Turn a vector representation of lower triangular matrix to connectome.
    
    a = vector
    dim = dimension of connectome
    
    Returns:
    dim x dim connectome
    """
    A = np.zeros((dim,dim))
    mask = np.tri(dim,dtype=bool, k=0)
    A[mask]=a
    B = np.array(A).transpose()
    np.fill_diagonal(B,0)
    return A + B

def strat_mask(pheno,case,control,stratify = 'SITE',seed=0):
    """ Get a mask for pheno with all the specified cases & controls matched
        for the `stratify` category.
    
    Parameters
    ----------
    pheno: DataFrame
    case: str
    control: str
    stratify: str, default='SITE'
    seed: int, default=0

    Returns
    -------
    bool array
    """
    random.seed(seed)

    # Get all the cases
    case_mask = pheno.index.isin(pheno[pheno[case]==1].index)

    # Find the number of cases per strat category
    strat = pheno[pheno[case]==1][stratify].unique()
    strat_dict = pheno[pheno[case]==1].groupby(stratify).count()[case].to_dict()

    # Randomly select the same number of controls from each category
    # if there aren't enough controls take them all
    control_mask = np.zeros(pheno.shape[0],dtype=bool)
    for s in strat:
        n_con = strat_dict[s]
        controls = pheno[(pheno[control]==1)&(pheno[stratify] == s)].index.to_list()
        if (n_con > len(controls)):
            n_con = len(controls)
        idx = random.sample(controls,k=n_con)
        control_mask = control_mask + pheno.index.isin(idx)
        
    return case_mask + control_mask

class caseControlDataset(Dataset):
    def __init__(self,case,pheno_path,conn_path,format=1,seed=0):
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
        format: int
            0: vector of 2080
            1: shuffled 2d array 40x52
            2: connectome of 1x64x64
        seed: int
            Seed to fix the random shuffle of the vector into 2d array.
        """
        assert format in [0,1,2]
        self.format = format
        self.seed = seed
        self.name = case

        control = 'non_carriers'
        if case in ['SZ','ADHD','BIP','ASD']:
            control = 'CON_IPC'
        self.conn_path = os.path.join(conn_path,'connectome_{}_cambridge64.npy')

        pheno = pd.read_csv(pheno_path,index_col=0)

        subject_mask = strat_mask(pheno,case,control,seed=seed)
        
        self.idx = pheno[subject_mask].index

        self.Y = pheno.loc[self.idx][case].values.astype(int)

        del pheno

    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self,idx):
        """
        Returns
        -------
        X: array
            Either a vector or randomly shuffled 2d array or connectome.
        Y: dict[str:array]
            Dictionary from dataset name to labels (array of int).
        """
        sub_id = self.idx[idx]
        conn = np.load(self.conn_path.format(sub_id))#[mask]
        mask = np.tri(64,dtype=bool)

        if self.format == 0:
            return conn[mask], {self.name:self.Y[idx]}
        elif self.format == 1:
            np.random.seed(self.seed)
            return conn[mask][np.random.permutation(2080)].reshape(40,52), {self.name:self.Y[idx]}
        else:
            return torch.unsqueeze(torch.from_numpy(conn),0), {self.name:self.Y[idx]}

class ukbbSexDataset(Dataset):
    def __init__(self,pheno_path,conn_path,dim=2,seed=0):
        assert ((dim == 1)|(dim == 2))
        self.dim = dim
        self.seed = seed
        self.name = 'ukbb_sex'

        conn_path = os.path.join(conn_path,'connectome_{}_cambridge64.npy')
        pheno = pd.read_csv(pheno_path,index_col=0)

        idx = pheno[(pheno['PI']=='UKBB') & (pheno['non_carriers']==1)].index
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
