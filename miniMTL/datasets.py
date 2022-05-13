import pandas as pd
import numpy as np
import random
import os

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

SEED = 123

def strat_mask(pheno,case,control,stratify = 'SITE',seed=None):
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
    if seed is not None:
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

def get_connectome(sub_id,conn_path,format):
    """ Load & format connectome for given sub_id. 
    See caseControlDataset class for format description.

    Parameters
    ----------
    sub_id: str
        Subject id.
    conn_path: str
        Path to connectome .npy file.
    format: int
        Which format (in [0,1,2]).
    
    Raises
    ------
    ValueError
        Connectome format (int encoded) must be in [0,1,2].

    Returns
    -------
    array or Tensor
        Connectome in requested format.
    """
    conn = np.load(conn_path.format(sub_id))
    mask = np.tri(64,dtype=bool)

    if format == 0:
        return conn[mask]
    elif format == 1:
        np.random.seed(SEED)
        return conn[mask][np.random.permutation(2080)].reshape(40,52)
    elif format == 2:
        return torch.unsqueeze(torch.from_numpy(conn),0)
    else:
        raise ValueError('Connectome format (int encoded) must be in [0,1,2].')

def get_concat(conf,sub_id,conn_path,format):
    """ Load & format concatenated confounds and connectome for given sub_id. 
    See caseControlDataset class for format description.

    Parameters
    ----------
    conf: array
        Vector of confounds.
    sub_id: str
        Subject id.
    conn_path: str
        Path to connectome .npy file.
    format: int
        Which format (in [0,1,2]).
    
    Raises
    ------
    ValueError
        Concatenated conn + conf format (int encoded) must be in [0,1].

    Returns
    -------
    array or Tensor
        Concatenated confounds and connectome in requested format.
    """
    conn = np.load(conn_path.format(sub_id))
    mask = np.tri(64,dtype=bool)
    concat = np.concatenate([conn[mask],conf])

    if format == 0:
        return concat
    elif format == 1:
        np.random.seed(SEED)
        return np.pad(concat[np.random.permutation(2080+58)],2).reshape(42,51)
    else:
        raise ValueError('Concatenated conn + conf format (int encoded) must be in [0,1].')

class caseControlDataset(Dataset):
    """ Generate a case control dataset for the given case according to the desired parameters.

    Parameters
    ----------
    case: str
        Case label.
    pheno_path: str
        Path to pheno .csv file.
    id_path: str, default=None
        Path to balanced dataset ids .csv file (needed for 'balanced' strategy).
    conn_path: str, default=None
        Path to connectome dir (needed for 'conn' or 'concat' type).
    type: str, default='concat'
        Which type of data to use, must be in ['conf','conn','concat']:
            - conf: one-hot encoded confounds (['AGE','SEX','SITE','mean_conn', 'FD_scrubbed']) (58)
            - conn: connectome (2080)
            - concat: concatenated conf + conn (2080 + 58)
    strategy: str, default='balanced'
        Which type of dataset to use, must be in ['balanced','stratified']
            - balanced: fixed datasets & CV splits generated to have balanced test sets.
            - stratified: random sample of controls matched by site.
    format: int, default=0
        Which format to return the data in (must be in [0,1,2] depending on type):
            - 0: vector (58, 2080, 2080 + 58)
            - 1: random shuffle of vector into array (N/A, 40x52, 42x51)
            - 2: full connectome (N/A, N/A, 64x64)
    """
    def __init__(self,case,pheno_path,id_path=None,conn_path=None,type='concat',strategy='balanced',format=0):
        assert type in ['concat','conn','conf']
        assert strategy in ['balanced','stratified']
        self.name = case
        self.type = type
        self.strategy = strategy
        self.format = format
        if conn_path:
            self.conn_path = os.path.join(conn_path,'connectome_{}_cambridge64.npy')
        pheno = pd.read_csv(pheno_path,index_col=0)

        # Select subjects
        if self.strategy == 'balanced':
            if id_path is None:
                raise ValueError("Must specify id_path for 'balanced' strategy.")
            self.ids = pd.read_csv(os.path.join(id_path,f"{case}.csv"),index_col=0)
            self.idx = self.ids.index
        elif self.strategy == 'stratified':
            control = 'CON_IPC' if case in ['SZ','BIP','ASD'] else 'non_carriers'
            subject_mask = strat_mask(pheno,case,control)
            self.idx = pheno[subject_mask].index
        
        # Get confounds if needed
        if self.type != 'conn':
            confounds = ['AGE','SEX','SITE','mean_conn', 'FD_scrubbed']
            p = pd.get_dummies(pheno[confounds],['SEX','SITE'])
            cols = ['AGE','mean_conn', 'FD_scrubbed'] + [c for c in p.columns if 'SEX' in c ] + [c for c in p.columns if 'SITE' in c ]
            p = p[p.index.isin(self.idx)]
            self.X_conf = p[cols]

            # Cleanup
            del p
        
        # Get labels
        self.Y = pheno.loc[self.idx][case].values.astype(int)

        # Cleanup
        del pheno

    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self,idx):
        if self.type == 'conn':
            conn = get_connectome(self.idx[idx], self.conn_path,self.format)
            return conn, {self.name:self.Y[idx]}
        elif self.type == 'conf':
            if self.format != 0:
                raise Warning('Confound format can only be 0 (vector).')
            return self.X_conf.iloc[idx].values, {self.name:self.Y[idx]}
        elif self.type == 'concat':
            concat = get_concat(self.X_conf.iloc[idx],self.idx[idx], self.conn_path,self.format)
            return concat, {self.name:self.Y[idx]}
    
    def split_data(self,random=True,fold=0,splits=(0.8,0.2),seed=None):
        """ Split dataset into train & test sets.

        Parameters
        ----------
        random: bool, default=True
            Wether to use random splits, otherwise use balanced test sets 
            (only compatible with 'balanced' strategy).
        fold: int, default=0
            If using 'balanced', which fold of balanced test sets to use.
        splits: tuple, default=(0.8,0.2)
            If using random splits, proportion of training & test data respectively.
        seed: int, default=None
            If using random splits, fix the random state.

        Returns
        -------
        train_idx, test_idx
        """
        if not random:
            if self.strategy != 'balanced':
                raise ValueError("Balanced CV folds only available for balanced dataset (set strategy to 'balanced').")
            rr = np.array(range(len(self.idx)))
            train_idx = rr[self.ids[f"fold_{fold}"] == 0]
            test_idx = rr[self.ids[f"fold_{fold}"] == 1]
        else:
            train_idx, test_idx, _, _ = train_test_split(range(len(self.idx)),
                                                    self.Y,
                                                    stratify=self.Y,
                                                    test_size=splits[1],
                                                    random_state=seed)
        return train_idx, test_idx

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