import torch
from torch.utils.data import DataLoader

from miniMTL.datasets import *
from miniMTL.models import *
from miniMTL.util import split_data

"""
March 4 2022
    Working single task usage.
March 9 2022
    BROKEN - update for new formats
"""

# From torch fundamentals course
def train(dataloader, model, loss_fn, name, optimizer,device='cpu'):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y[name].to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# From torch fundamentals course
def test(dataloader, model,loss_fn,name,device='cpu'):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y[name].to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    p_pheno = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/pheno_01-12-21.csv'
    p_conn = '/home/harveyaa/Documents/fMRI/data/ukbb_9cohorts/connectomes/'

    data = ukbbSexDataset(p_pheno,p_conn)
    case = 'ukbb_sex'#SZ
    #data = caseControlDataset('SZ',p_pheno,p_conn)

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
        train(trainloader, model, loss_fn, case, optimizer)
        test(testloader, model, loss_fn, case)
    print("Done!")