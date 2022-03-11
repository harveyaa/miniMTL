import torch
from torch.utils.data import DataLoader

from miniMTL.datasets import caseControlDataset
from miniMTL.models import *
from miniMTL.training import Trainer
from miniMTL.util import split_data

""" 
HPS v1
------
March 11 2022
    Matched multi-input training strategy from snorkel.

    TODO:
    - Manage devices (model & data)
    - Make datasets less heavy & foolish
"""
# Define paths to data
p_pheno = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/pheno_01-12-21.csv'
p_conn = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/connectomes_01-12-21.csv'

# Create Datasets
case1 = 'SZ'
case2 = 'BIP'

data1 = caseControlDataset(case1,p_pheno,p_conn)
data2 = caseControlDataset(case2,p_pheno,p_conn)

# Map task name to match dataset name
task_to_name = {'task1':case1,'task2':case2}

# Split data into train & test
train_data1, test_data1 = split_data(data1)
train_data2, test_data2 = split_data(data2)

# Define DataLoaders
trainloader1 = DataLoader(train_data1, batch_size=16, shuffle=True)
testloader1 = DataLoader(test_data1, batch_size=16, shuffle=True)

trainloader2 = DataLoader(train_data2, batch_size=16, shuffle=True)
testloader2 = DataLoader(test_data2, batch_size=16, shuffle=True)

# Group DataLoaders into train & test with task labels
trainloaders = {task_to_name['task1']:trainloader1, task_to_name['task2']:trainloader2}
testloaders = {task_to_name['task1']:testloader1, task_to_name['task2']:testloader2}

# Create dictionary of loss functions
loss_fns = {task_to_name['task1']:nn.CrossEntropyLoss(),task_to_name['task2']:nn.CrossEntropyLoss()}

# Initiate the model
model = HPSModel(encoder().double(),
                {task_to_name['task1']:head().double(),task_to_name['task2']:head().double()},
                loss_fns)

# Create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
trainer = Trainer(optimizer,50)
trainer.fit(model,trainloaders,testloaders)

# Evaluate the model
print(model.score(testloaders))