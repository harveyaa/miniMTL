
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

def split_data(dataset,test_size=0.2,seed=0):
    train_idx, test_idx, _, _ = train_test_split(range(len(dataset)),
                                                dataset.Y,
                                                stratify=dataset.Y,
                                                test_size=test_size,
                                                random_state=seed)

    return Subset(dataset,train_idx), Subset(dataset,test_idx)

# From torch fundamentals course
def train(dataloader, model, loss_fn, optimizer,device='cpu'):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
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
def test(dataloader, model,loss_fn,device='cpu'):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_step(X,y,model,loss_fn,optimizer,task,device='cpu'):
    X, y = X.to(device), y.to(device)
            
    # Compute prediction error
    pred = model(X,task)
    loss = loss_fn(pred, y)
            
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
