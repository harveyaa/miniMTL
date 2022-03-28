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

class encoder0(nn.Module):
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

class encoder1(nn.Module):
    """Plus a FC layer."""
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

class encoder2(nn.Module):
    """Plus a conv layer."""
    def __init__(self):
        super().__init__()
        # in_channels, out_channels
        self.conv1 = nn.Conv2d(1, 256, (40,1))
        self.batch01 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 64, (1,10))
        self.batch02 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*43, 64)
        self.batch1 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout()
    
    def forward(self,x):
        x = self.conv1(torch.unsqueeze(x,dim=1))
        x = self.batch01(x)
        x = self.conv2(x)
        x = self.batch02(x)
        x = x.view(x.size()[0],-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.batch1(x)
        return x

class head0(nn.Module):
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

class head1(nn.Module):
    """Minus a FC layer."""
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(64,64)
        self.batch3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,2)

        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.batch3(x)
        x = self.softmax(self.fc4(x))
        return x

class head2(nn.Module):
    """Plus a FC layer."""
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(64,64)
        self.batch3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,64)
        self.batch4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64,32)
        self.batch5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32,2)

        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.batch3(x)
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.batch4(x)
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.batch5(x)
        x = self.softmax(self.fc6(x))
        return x