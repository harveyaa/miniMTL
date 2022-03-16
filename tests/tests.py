import os
import numpy as np
import pandas as pd
from miniMTL.datasets import *
from miniMTL.util import *
from miniMTL.training import *
from torch.utils.data import DataLoader

def gen_connectomes(dir,n_cases=10,seed=0):
    """Put fake connectomes in a tmpdir to test dataset creation."""
    np.random.seed(seed)

    for i in range(2*n_cases):
        conn = np.random.randn(64,64)
        np.save(os.path.join(dir,f"connectome_{i}_cambridge64.npy"),conn)

def gen_pheno(dir,n_cases=10,n_tasks=2,seed=0):
    """Put fake pheno in a tmpdir to test dataset creation."""
    np.random.seed(seed)

    labels_case = np.ones((n_cases,n_tasks)) #np.random.binomial(1,0.5,size=(n_cases,n_tasks))
    labels_con = np.ones((n_cases,1))
    df_case = pd.DataFrame(labels_case,columns = [f"task{i}" for i in range(n_tasks)])
    df_con = pd.DataFrame(np.ones((n_cases,1)),index=range(n_cases,2*n_cases),columns=['non_carriers'])
    pheno = pd.concat([df_case,df_con]).fillna(0)
    pheno['PI'] = 'study0'
    pheno.to_csv(os.path.join(dir,'pheno.csv'))

def gen_dataset(dir,n_cases=10,n_tasks=2,seed=0):
    """
    Returns
    -------
    dict[str:caseControlDataset]
        Labels are 'task{i}' for i in range(n_tasks).
    """
    gen_connectomes(dir,n_cases=n_cases,seed=seed)
    gen_pheno(dir,n_cases=n_cases,n_tasks=n_tasks,seed=seed)
    p_pheno = os.path.join(dir,'pheno.csv')

    data = {}
    for i in range(n_tasks):
        data[f'task{i}'] = caseControlDataset(f'task{i}',p_pheno,dir)
    return data


class TestData:
    def test_case_control_dataset(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task1',p_pheno,tmpdir)
        assert data.name == 'task1'

class TestTraining:
    def test_get_batches_shuffled(self,tmpdir):
        data = gen_dataset(tmpdir,n_cases=20)
        
        trainloaders = {}
        for k in data.keys():
            train_d, _ = split_data(data[k])
            trainloaders[k] = DataLoader(train_d, batch_size=16, shuffle=True)
        
        tasks = []
        for X,Y_dict in get_batches(trainloaders,shuffle=True):
            tasks.append(list(Y_dict.keys())[0])
        
        assert tasks == ['task1', 'task0', 'task0', 'task1']

    def test_get_batches_sequential(self,tmpdir):
        data = gen_dataset(tmpdir,n_cases=20)
        
        trainloaders = {}
        for k in data.keys():
            train_d, _ = split_data(data[k])
            trainloaders[k] = DataLoader(train_d, batch_size=16, shuffle=True)
        
        tasks = []
        for X,Y_dict in get_batches(trainloaders,shuffle=False):
            tasks.append(list(Y_dict.keys())[0])
        
        assert tasks == ['task0', 'task0', 'task1', 'task1']
