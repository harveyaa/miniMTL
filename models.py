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
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.batch2(x)
        return x

class head(nn.Module):
    def __init__(self):
        super().__init__()
        #self.fc1 = nn.Linear(256*52, 64)
        #self.batch1 = nn.BatchNorm1d(64)
        #self.fc2 = nn.Linear(64, 64)
        #self.batch2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64,64)
        self.batch3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,2)

        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        #x = x.view(x.size()[0],-1)
        #x = self.dropout(F.relu(self.fc1(x)))
        #x = self.batch1(x)
        #x = self.dropout(F.relu(self.fc2(x)))
        #x = self.batch2(x)
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.batch3(x)
        x = self.softmax(self.fc4(x))
        return x

class HPSModel(nn.Module):
    """ Multi-input HPS."""
    def __init__(self,encoder,decoders,loss_fns):
        """
        Parameters
        ----------
        encoder: nn.Module
        decoders: dict str: nn.Module
        loss_fns: dict str: loss fn
        """
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.loss_fns = loss_fns
    
    def forward(self,X,task_names):
        """
        Parameters
        ----------
        X: minibatch
        task_names: list of tasks x goes to

        Returns
        -------
        dict: task_name: output
        """
        X = self.encoder(X)
        outputs = {}
        for task in task_names:
            outputs[task] = self.decoders[task](X)
        return outputs
    
    def calculate_loss(self,X,Y_dict):
        """
        Parameters
        ----------
        X: batch
        Y_dict: dict: str labels
        """
        task_names = Y_dict.keys()
        outputs = self.forward(X,task_names)
        losses = {}
        for task in task_names:
            losses[task] = self.loss_fns[task](outputs[task],Y_dict[task])
        return losses