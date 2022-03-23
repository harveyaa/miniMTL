import os
import numpy as np
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader

from miniMTL.datasets import *
from miniMTL.util import *
from miniMTL.models import *
from miniMTL.training import *
from miniMTL.logging import *
from miniMTL.hps import HPSModel

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
    pheno['SITE'] = 'site0'
    pheno.to_csv(os.path.join(dir,'pheno.csv'))

def gen_case_con_dataset(dir,n_cases=10,n_tasks=2,seed=0):
    """
    Generate a caseControlDataset.

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

def gen_model_and_loaders(dataset,batch_size=16,shuffle=True):
    """
    Generate a model and dataloaders.

    Parameters
    ----------
    batch_size: int, default=16
        batch_size for DataLoaders
    shuffle: bool
        Wether or not to shuffle DataLoaders.

    Returns
    -------
    HPSModel
        Model built from dataset using default encoder & head modules.
    DataLoader
        Training dataloader.
    DataLoader
        Test dataloader.
    """
    trainloaders = {}
    testloaders = {}
    loss_fns = {}
    decoders = {}
    for k in dataset.keys():
        train_d, test_d = split_data(dataset[k])

        trainloaders[k] = DataLoader(train_d, batch_size=batch_size, shuffle=shuffle)
        testloaders[k] = DataLoader(test_d, batch_size=batch_size, shuffle=shuffle)

        loss_fns[k] = nn.CrossEntropyLoss()
        decoders[k] = head0().double()
    
    model = HPSModel(encoder0().double(),
                    decoders,
                    loss_fns)
    
    return model, trainloaders, testloaders

class TestData:
    def test_case_control_dataset(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task1',p_pheno,tmpdir)
        assert data.name == 'task1'
    
    def test_dataloader(self,tmpdir):
        data = gen_case_con_dataset(tmpdir,n_cases=100)
        _, trainloaders, testloaders = gen_model_and_loaders(data,shuffle=False)
        trainloader = trainloaders[list(trainloaders.keys())[0]]
        X, Y_dict = next(iter(trainloader))
        assert isinstance(X,torch.Tensor)
        assert isinstance(Y_dict,dict)

class TestModel:
    def test_init(self,tmpdir):
        """ Test designed to make sure 'model' tracks the encoder and all the decoders."""
        n_tasks = 5
        data = gen_case_con_dataset(tmpdir,n_cases=100,n_tasks=n_tasks)
        model, _,_ = gen_model_and_loaders(data,shuffle=False)

        # Check that all the task heads are in the model
        names = []
        for name, param in model.named_parameters():
            names.append(name.split('.')[0])
        assert len(np.unique(names)) == n_tasks + 1
        
    def test_forward(self,tmpdir):
        data = gen_case_con_dataset(tmpdir,n_cases=100)
        model, trainloaders, testloaders = gen_model_and_loaders(data,shuffle=False)
        trainloader = trainloaders[list(trainloaders.keys())[0]]
        X, Y_dict = next(iter(trainloader))
        task = list(Y_dict.keys())[0]
        pred = model.forward(X.to(model.device),[task])[task]
        assert isinstance(pred,torch.Tensor)

    def test_score(self,tmpdir):
        data = gen_case_con_dataset(tmpdir,n_cases=100)
        model, trainloader, testloader = gen_model_and_loaders(data,shuffle=False)
        m = model.score(testloader)
        assert isinstance(m,dict)
    
    def test_device(self,tmpdir):
        data = gen_case_con_dataset(tmpdir,n_cases=100)
        model, trainloader, testloader = gen_model_and_loaders(data,shuffle=False)
        assert torch.cuda.is_available() == (next(model.parameters()).is_cuda)

class TestLogging:
    def test_logging(self,tmpdir):
        logger = Logger(log_dir=tmpdir)

        for i in range(10):
            logger.add_scalar('task1','accuracy',np.random.randn(),i)
            logger.add_scalar('task1','loss',np.random.randn(),i)
            logger.add_scalar('task2','loss',np.random.randn(),i)
        
        assert logger.log_dir == tmpdir
        logger.save()

        df = pd.read_csv(os.path.join(tmpdir,logger.filename),header=[0,1],index_col=0)
        assert df.columns.to_list() == [('task1', 'accuracy'), ('task1', 'loss'), ('task2', 'loss')]

class TestTraining:
    def test_get_batches_shuffled(self,tmpdir):
        data = gen_case_con_dataset(tmpdir,n_cases=20)
        
        trainloaders = {}
        for k in data.keys():
            train_d, _ = split_data(data[k])
            trainloaders[k] = DataLoader(train_d, batch_size=16, shuffle=True)
        
        tasks = []
        for X,Y_dict in get_batches(trainloaders,shuffle=True):
            tasks.append(list(Y_dict.keys())[0])
        
        print(tasks)
        assert tasks == ['task0', 'task1', 'task1', 'task0']

    def test_get_batches_sequential(self,tmpdir):
        data = gen_case_con_dataset(tmpdir,n_cases=20)
        
        trainloaders = {}
        for k in data.keys():
            train_d, _ = split_data(data[k])
            trainloaders[k] = DataLoader(train_d, batch_size=16, shuffle=True)
        
        tasks = []
        for X,Y_dict in get_batches(trainloaders,shuffle=False):
            tasks.append(list(Y_dict.keys())[0])
        
        assert tasks == ['task0', 'task0', 'task1', 'task1']
