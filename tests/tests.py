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
    pheno['FD_scrubbed'] = 0.5
    pheno['mean_conn'] = 0.3
    pheno['SEX'] = 'Male'
    pheno['AGE'] = 30
    pheno.to_csv(os.path.join(dir,'pheno.csv'))

def gen_cv_ids(dir,name='task0',n_subs=20, seed=0):
    """Put fake cv fold ids csv file in a tmpdir to test dataset creation."""
    np.random.seed(seed)
    df = pd.DataFrame(np.random.binomial(1,0.3,size=(n_subs,6)), columns = [f'fold_{i}' for i in range(5)]+[name])
    df.to_csv(os.path.join(dir,f'{name}.csv'))

def gen_case_con_dataset(dir,n_cases=10,n_tasks=2,seed=0,format=1):
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
        data[f'task{i}'] = caseControlDataset(f'task{i}',p_pheno,conn_path=dir,type='conn',strategy='stratified',format=format)
    return data

def gen_model_and_loaders(dataset,batch_size=16,shuffle=True,model=0):
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
        decoders[k] = eval(f'head{model}().double()')
    
    model = HPSModel(eval(f'encoder{model}().double()'),
                    decoders,
                    loss_fns)
    
    return model, trainloaders, testloaders

# NO DROPOUT FOR STABLE TESTS
class encoder999(nn.Module):
    def __init__(self,dim=2080,width=10):
        super().__init__()
        # in_channels, out_channels
        self.fc1 = nn.Linear(dim, width)
        self.fc2 = nn.Linear(width, width)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class head999(nn.Module):
    def __init__(self,width=10):
        super().__init__()
        self.fc3 = nn.Linear(width,width)
        self.fc4 = nn.Linear(width,2)
    
    def forward(self,x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class TestData:
    def test_cc_dataset(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task1',p_pheno,conn_path=tmpdir,type='conn',strategy='stratified')
        assert data.name == 'task1'
    
    def test_cc_dataset_0(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task1',p_pheno,conn_path=tmpdir,type='conn',strategy='stratified',format=0)
        assert data.__getitem__(0)[0].shape[0] == 2080
    
    def test_cc_dataset_1(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task1',p_pheno,conn_path=tmpdir,type='conn',strategy='stratified',format=1)
        assert data.__getitem__(0)[0].shape == (40,52)
    
    def test_cc_dataset_2(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task1',p_pheno,conn_path=tmpdir,type='conn',strategy='stratified',format=2)
        assert data.__getitem__(0)[0].shape == (1,64,64)
    
    def test_cc_dataset_balanced(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        gen_cv_ids(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task0',p_pheno,id_path=tmpdir,conn_path=tmpdir,type='conn',strategy='balanced')
        assert data.name == 'task0'
    
    def test_cc_balanced_split_data(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        gen_cv_ids(tmpdir)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        ids = pd.read_csv(os.path.join(tmpdir,"task0.csv"),index_col=0)
        data = caseControlDataset('task0',p_pheno,id_path=tmpdir,conn_path=tmpdir,type='conn',strategy='balanced')
        for i in range(5):
            train_idx,test_idx = data.split_data(random=False,fold=i)
            assert len(set(ids[ids[f'fold_{i}'] == 1].index.to_list()).difference(set(test_idx))) == 0
            assert len(set(ids[ids[f'fold_{i}'] == 0].index.to_list()).difference(set(train_idx))) == 0
    
    def test_balanced_confounds_dataset(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        gen_cv_ids(tmpdir,name='task0')
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task0',p_pheno,id_path=tmpdir,type='conf',strategy='balanced')
        assert data.name == 'task0'
    
    def test_balanced_confounds_split_data(self,tmpdir):
        gen_connectomes(tmpdir)
        gen_pheno(tmpdir)
        gen_cv_ids(tmpdir,name='task0')
        ids = pd.read_csv(os.path.join(tmpdir,"task0.csv"),index_col=0)
        p_pheno = os.path.join(tmpdir,'pheno.csv')
        data = caseControlDataset('task0',p_pheno,id_path=tmpdir,type='conf',strategy='balanced')
        for i in range(5):
            train_idx,test_idx = data.split_data(random=False,fold=i)
            assert len(set(ids[ids[f'fold_{i}'] == 1].index.to_list()).difference(set(test_idx))) == 0
            assert len(set(ids[ids[f'fold_{i}'] == 0].index.to_list()).difference(set(train_idx))) == 0
    
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
    
    def test_hps_vs_chain(self,tmpdir):
        """ Test to check if net output changes once it's a part of HPS model."""
        data = gen_case_con_dataset(tmpdir,n_cases=100,format=0)
        trainloaders = {}
        testloaders = {}
        loss_fns = {}
        decoders = {}
        for k in data.keys():
            train_d, test_d = split_data(data[k])

            trainloaders[k] = DataLoader(train_d, batch_size=16, shuffle=True)
            testloaders[k] = DataLoader(test_d, batch_size=16, shuffle=True)

            loss_fns[k] = nn.CrossEntropyLoss()
            decoders[k] = head999().double()
        
        encoder = encoder999().double()
        model = HPSModel(encoder,
                        decoders,
                        loss_fns)
        
        task = 'task0'
        X, _ = next(iter(trainloaders[task]))
 
        pred_hps = model.forward(X.to(model.device),[task])[task]
        pred_chain = decoders[task](encoder(X))

        assert (pred_chain == pred_hps).sum() == 2*pred_hps.size(0)
    
    def test_hps_vs_chain2(self,tmpdir):
        """ Thorough test to check if net output changes once it's a part of HPS model."""
        data = gen_case_con_dataset(tmpdir,n_cases=100,format=0)
        trainloaders = {}
        testloaders = {}
        loss_fns = {}
        decoders = {}
        for k in data.keys():
            train_d, test_d = split_data(data[k])

            trainloaders[k] = DataLoader(train_d, batch_size=16, shuffle=True)
            testloaders[k] = DataLoader(test_d, batch_size=16, shuffle=True)

            loss_fns[k] = nn.CrossEntropyLoss()
            dec = head999().double()
            torch.nn.init.constant_(dec.fc3.weight,0.5)
            torch.nn.init.constant_(dec.fc3.bias,0.5)
            torch.nn.init.constant_(dec.fc4.weight,0.5)
            torch.nn.init.constant_(dec.fc4.bias,0.5)
            decoders[k] = dec
        
        encoder = encoder999().double()
        torch.nn.init.constant_(encoder.fc1.weight,0.5)
        torch.nn.init.constant_(encoder.fc1.bias,0.5)
        torch.nn.init.constant_(encoder.fc2.weight,0.5)
        torch.nn.init.constant_(encoder.fc2.bias,0.5)
        model = HPSModel(encoder,
                        decoders,
                        loss_fns)
        
        encoder2 = encoder999().double()
        dec2 = head999().double()
        torch.nn.init.constant_(encoder2.fc1.weight,0.5)
        torch.nn.init.constant_(encoder2.fc1.bias,0.5)
        torch.nn.init.constant_(encoder2.fc2.weight,0.5)
        torch.nn.init.constant_(encoder2.fc2.bias,0.5)
        torch.nn.init.constant_(dec2.fc3.weight,0.5)
        torch.nn.init.constant_(dec2.fc3.bias,0.5)
        torch.nn.init.constant_(dec2.fc4.weight,0.5)
        torch.nn.init.constant_(dec2.fc4.bias,0.5)
        
        task = 'task0'
        X, _ = next(iter(trainloaders[task]))
 
        pred_hps = model.forward(X.to(model.device),[task])[task]
        pred_chain = decoders[task](encoder(X))
        pred_out_chain = dec2(encoder2(X))

        #print('MLP 2080')
        #print('HPS')
        #print(pred_hps)
        #print('CHAIN HPS MODULES')
        #print(pred_chain)
        #print('CHAIN NEW SAME MODULES')
        #print(pred_out_chain)
        assert (pred_chain == pred_hps).sum() == 2*pred_hps.size(0)
        assert (pred_out_chain == pred_hps).sum() == 2*pred_hps.size(0)
    
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
        for X,Y_dict in get_batches(trainloaders,shuffle=True,seed=1):
            tasks.append(list(Y_dict.keys())[0])
        
        print(tasks)
        assert tasks == ['task1', 'task0', 'task1', 'task0']

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
