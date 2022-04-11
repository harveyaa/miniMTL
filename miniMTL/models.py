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


# NICOLAS FARRUGIA
class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''
    def __init__(self, in_planes, planes,example,bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes,planes,(1,self.d),bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes,planes,(self.d,1),bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d,3)+torch.cat([b]*self.d,2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(BrainNetCNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        
        self.e2econv1 = E2EBlock(1,32,example,bias=True)
        self.e2econv2 = E2EBlock(32,64,example,bias=True)
        self.E2N = torch.nn.Conv2d(64,1,(1,self.d))
        self.N2G = torch.nn.Conv2d(1,256,(self.d,1))
        self.dense1 = torch.nn.Linear(256,128)
        self.dense2 = torch.nn.Linear(128,30)
        self.dense3 = torch.nn.Linear(30,2)
        
    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out),negative_slope=0.33) 
        out = F.leaky_relu(self.E2N(out),negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out),negative_slope=0.33),p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out),negative_slope=0.33),p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out),negative_slope=0.33),p=0.5)
        out = F.leaky_relu(self.dense3(out),negative_slope=0.33)
        
        return out