
import torch
import numpy as np

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

def trainMTL(dataloaders, model, loss_fns, optimizer,device='cpu'):
    tasks = list(dataloaders.keys())
    n_batches = dict(zip(tasks,[len(x) for x in list(dataloaders.values())]))
    max_batches = np.max(list(n_batches.values()))
    
    for i in range(max_batches):
        for task in tasks:
            dataloader = dataloaders[task]

            if (max_batches - n_batches[task]) >= i:
                X,y = next(iter(dataloader))

                train_step(X,y,model,loss_fns[task],optimizer,task,device=device)

# From torch fundamentals course
def testMTL(dataloaders, model,loss_fns,device='cpu'):
    tasks = list(dataloaders.keys())

    model.eval()
    for task in tasks:
        dataloader = dataloaders[task]
        loss_fn = loss_fns[task]

        size = len(dataloader.dataset)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X,task)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(task)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")