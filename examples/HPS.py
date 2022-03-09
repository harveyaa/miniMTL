import torch
from torch.utils.data import DataLoader

from datasets import *
from models import *
from util import train,test

""" 
HPS v0
------
March 4 2022
    Functional as in produces no errors but logically potentially unsound.
March 9 2022
    BROKEN - update for new formats
"""

p_pheno = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/pheno_01-12-21.csv'
p_conn = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/connectomes_01-12-21.csv'

data1 = caseControlDataset('SZ',p_pheno,p_conn)
data2 = caseControlDataset('BIP',p_pheno,p_conn)

train_data1, test_data1 = split_data(data1)
train_data2, test_data2 = split_data(data2)

trainloader1 = DataLoader(train_data1, batch_size=16, shuffle=True)
testloader1 = DataLoader(test_data1, batch_size=16, shuffle=True)

trainloader2 = DataLoader(train_data2, batch_size=16, shuffle=True)
testloader2 = DataLoader(test_data2, batch_size=16, shuffle=True)

model = HPSModel(encoder().double(),{'task1':head().double(),'task2':head().double()})

trainloaders = {'task1':trainloader1,'task2':trainloader2}
testloaders = {'task1':testloader1, 'task2':testloader2}

loss_fns = {'task1':nn.CrossEntropyLoss(),'task2':nn.CrossEntropyLoss()}
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = 'cpu'

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainMTL(trainloaders, model, loss_fns, optimizer)
    testMTL(testloaders, model, loss_fns)
print("Done!")