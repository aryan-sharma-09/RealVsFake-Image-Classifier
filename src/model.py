import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3)

        self.relu=nn.ReLU()
        
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)

        self.flatten=nn.Flatten()

        self.fc=nn.Linear(32*62*62,2)

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)


        x=self.flatten(x)

        x=self.fc(x)

        return x