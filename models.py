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
    def __init__(self,encoder,decoders):
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
    
    def forward(self,x,task):
        x = self.encoder(x)
        x = self.decoders[task](x)
        return x