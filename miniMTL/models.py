import torch
import torch.nn as nn
import torch.nn.functional as F

class ukbbSex(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels, out_channels
        self.conv = nn.Conv2d(1, 256, (40,1))
        self.batch0 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*52, 64)
        self.batch1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.batch2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64,2)

        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.conv(torch.unsqueeze(x,dim=1))
        x = self.batch0(x)
        x = x.view(x.size()[0],-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.batch1(x)
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.batch2(x)
        x = self.softmax(self.fc3(x))
        return x

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels, out_channels
        self.conv = nn.Conv2d(1, 256, (40,1))
        self.batch0 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*52, 64)
        self.batch1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.batch2 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout()
    
    def forward(self,x):
        x = self.conv(torch.unsqueeze(x,dim=1))
        x = self.batch0(x)
        x = x.view(x.size()[0],-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.batch1(x)
        #x = self.dropout(F.relu(self.fc2(x)))
        #x = self.batch2(x)
        return x

class head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(64,64)
        self.batch3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,64)
        self.batch4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64,2)

        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.batch3(x)
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.batch4(x)
        x = self.softmax(self.fc5(x))
        return x

class HPSModel(nn.Module):
    """ Multi-input HPS."""
    def __init__(self,encoder,decoders,loss_fns, device='cpu'):
        """
        Parameters
        ----------
        encoder: nn.Module
            Shared portion of the model.
        decoders: dict[str, nn.Module]
            Dictionary from task name to task-specific decoder (head).
        loss_fns: dict[str, loss_fn]
            Dictionary from task name to torch loss function.
        """
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.loss_fns = loss_fns

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(torch.device(self.device))
        print(f'Initialized HPSModel using: {self.device}.')
    
    def forward(self,X,task_names):
        """
        Parameters
        ----------
        X: Tensor
            A batch of data.
        task_names: list[str]
            List of task names associated with X.

        Returns
        -------
        dict[str, Tensor]
            Dictionary from task name to associated output.
        """
        X = self.encoder(X)
        print('made it ENCODER')
        outputs = {}
        for task in task_names:
            outputs[task] = self.decoders[task](X)
            print('made it TASK HEAD')
        return outputs
    
    def calculate_loss(self,X,Y_dict):
        """
        Parameters
        ----------
        X: Tensor
            A batch of data.
        Y_dict: dict[str, Tensor]
            Dictionary from task name to associated labels.
        
        Returns
        -------
        dict[str, Tensor]
            Dictionary from task name to associated loss.
        """
        task_names = Y_dict.keys()
        outputs = self.forward(X.to(self.device),task_names)
        losses = {}
        for task in task_names:
            losses[task] = self.loss_fns[task](outputs[task],Y_dict[task].to(self.device))
        return losses
    
    def score(self,dataloaders):
        """ 
        Score model on given data.

        Parameters
        ----------
        dataloaders: dict[str, DataLoader]
            Dictionary from task name to associated DataLoader.
        
        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary from task name to dictionary of metric label
            ('accuracy', 'test_loss') to value.
        """
        tasks = list(dataloaders.keys())
        self.eval()
        
        metrics = {}
        for task in tasks:
            dataloader = dataloaders[task]
            loss_fn = self.loss_fns[task]
            size = len(dataloader.dataset)

            test_loss, correct = 0, 0
            with torch.no_grad():
                i=0
                for X, Y_dict in dataloader:
                    X = X.to(self.device)                    
                    Y = Y_dict[task].to(self.device)
                    pred = self.forward(X,[task])[task]

                    test_loss += loss_fn(pred, Y).item()
                    correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
                    i +=1
            test_loss /= size
            correct /= size
            metrics[task] = {'accuracy':100*correct,'test_loss':test_loss}
        return metrics