import torch
from torch.utils.data import DataLoader

from datasets import *
from models import *
from util import train,test

"""
March 4 2022
    Working single task usage.
March 9 2022
    BROKEN - update for new formats
"""

p_pheno = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/pheno_01-12-21.csv'
p_conn = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/connectomes_01-12-21.csv'

#data = ukbbSexDataset(p_pheno,p_conn)
data = caseControlDataset('SZ',p_pheno,p_conn)

train_data, test_data = split_data(data)

trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
testloader = DataLoader(test_data, batch_size=16, shuffle=True)

model = ukbbSex().double()

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, model, loss_fn, optimizer)
    test(testloader, model, loss_fn)
print("Done!")