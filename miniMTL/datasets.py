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

def get_conf_mat(pheno,confounds):
    """ Reduce pheno to only confounds & onehot encode categorical columns.

    Parameters
    ----------
    pheno: DataFrame
        Phenotype DataFrame.
    confounds: list of str
        List of confounds to select.
    
    Returns:
    --------
    DataFrame
        Input DataFrame reduced to relevent (onehot encoded) columns.
    """
    categorical = [c for c in confounds if c in ['SEX','SITE']]
    p = pd.get_dummies(pheno[confounds],categorical)
    cols = confounds.copy()
    for cat in categorical:
        cols.remove(cat)
        cols += [c for c in p.columns if cat in c]
    return p[cols]

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
        if conf.shape[0] == 58:
            np.random.seed(SEED)
            return np.pad(concat[np.random.permutation(2080+58)],2).reshape(42,51)
        else:
            raise ValueError('Concatenated conn + conf_no_site format (int encoded) must be in 0.')
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
            - conf_no_site: one-hot encoded confounds (['AGE','SEX','mean_conn', 'FD_scrubbed']) (5)
            - concat_no_site: concatenated conf_no_site + conn (2080 + 5)
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
        assert type in ['concat','conn','conf','conf_no_site','concat_no_site']
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
        
        # Get confounds if needed (w/ SITE)
        if self.type != 'conn':
            if self.type in ['conf','concat']:
                confounds = ['AGE','SEX','SITE','mean_conn', 'FD_scrubbed']
            elif self.type in ['conf_no_site','concat_no_site']:
                confounds = ['AGE','SEX','mean_conn', 'FD_scrubbed']
            p = get_conf_mat(pheno,confounds)
            self.X_conf = p.loc[self.idx]

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
        elif (self.type == 'conf') | (self.type == 'conf_no_site'):
            if self.format != 0:
                raise Warning('Confound format can only be 0 (vector).')
            return self.X_conf.iloc[idx].values, {self.name:self.Y[idx]}
        elif (self.type == 'concat') | (self.type == 'concat_no_site'):
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

class confDataset(Dataset):
    """ Generate a confound dataset for the given site using control only according to the desired parameters.

    Parameters
    ----------
    site: str
        Site label.
    pheno_path: str
        Path to pheno .csv file.
    conf: str, default='SEX'
        Which confound to predict as a label, can be in ['SEX','AGE','FD_scrubbed']
    id_path: str, default=None
        Path to CV fold .csv file.
    conn_path: str, default=None
        Path to connectome dir (needed for 'conn' or 'concat' type).
    type: str, default='conn'
        Which type of data to use, must be in ['conf','conn','concat']:
            - conf: one-hot encoded confounds (['AGE','SITE','mean_conn', 'FD_scrubbed']) (56)
            - conn: connectome (2080)
            - concat: concatenated conf + conn (2080 + 56)
    format: int, default=0
        Which format to return the data in (must be in [0,1,2] depending on type):
            - 0: vector (58, 2080, 2080 + 58)
            - 1: random shuffle of vector into array (N/A, 40x52, 42x51)
            - 2: full connectome (N/A, N/A, 64x64)
    n_subsamp: int, default=None
        How many subjects to select in subsample (should only be relevant for UKBB).
    confounds: list of str, default=['AGE','SEX','SITE','mean_conn', 'FD_scrubbed']
        Which confounds to use as predictors (for type 'conf' or 'concat'). Note 'conf' (target variable)
        will always be removed even if it included in 'confounds'.
    """
    def __init__(self,site,pheno_path,conf='SEX',id_path=None,conn_path=None,type='conn',format=0,n_subsamp=50,controls_only=True,
                confounds = ['AGE','SEX','SITE','mean_conn', 'FD_scrubbed']):
        assert conf in ['SEX','AGE','FD_scrubbed']
        assert type in ['concat','conn','conf']
        # Sites with at least 30 controls
        assert site in ['ADHD1','ADHD3','ADHD5','ADHD6','HSJ','NYU','SZ1','SZ2','SZ3','SZ6','Svip1',
                        'Svip2','UCLA_CB','UCLA_DS1','UKBB11025','UKBB11026','UKBB11027','USM']
        self.name = site
        self.type = type
        self.format = format
        if conn_path:
            self.conn_path = os.path.join(conn_path,'connectome_{}_cambridge64.npy')
        pheno = pd.read_csv(pheno_path,index_col=0)

        # Select subjects
        if controls_only:
            if id_path is not None:
                # get IDS from .csv index
                if site[:4] != 'UKBB':
                    self.ids = pd.read_csv(os.path.join(id_path,f"{site}.csv"),index_col=0)
                else:
                    self.ids = pd.read_csv(os.path.join(id_path,f"{site}_{n_subsamp}.csv"),index_col=0)
                    
                self.idx = self.ids.index
            else:
                self.idx = pheno[(pheno['SITE']==site) & ((pheno['CON_IPC']==1)|(pheno['non_carriers']==1))].index
        else:
            self.idx = pheno[(pheno['SITE']==site)].index
        
        # Get confounds if needed
        if self.type != 'conn':
            confounds.remove(conf)
            p = get_conf_mat(pheno,confounds)
            self.X_conf = p.loc[self.idx]

            # Cleanup
            del p
        
        # Get response var
        if (conf == 'AGE') | (conf == 'FD_scrubbed'):
            self.Y = torch.from_numpy(pheno.loc[self.idx][conf].values).type(torch.double).unsqueeze(dim=1)
        elif conf == 'SEX':
            self.Y = pheno.loc[self.idx]['SEX'].map({'Female':1,'Male':0}).values.astype(int)

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
            Wether to use random splits, otherwise use premade non-overlapping CV folds.
        fold: int, default=0
            If using premade CV folds, which fold to use.
        splits: tuple, default=(0.8,0.2)
            If using random splits, proportion of training & test data respectively.
        seed: int, default=None
            If using random splits, fix the random state.

        Returns
        -------
        train_idx, test_idx
        """
        if not random:
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