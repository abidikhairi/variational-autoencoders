import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class VanillaAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaAE, self).__init__()
        
        self.encoder = MLP(input_size, hidden_size, output_size)
        self.decoder = MLP(output_size, hidden_size, input_size)


    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, code_size):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels, out_channels=16, kernel_size=3)
        self.fc = nn.Linear(16 * 3 * 3, code_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = F.relu(x)
        x = self.conv3(self.conv2(x))
        x = self.pool(x)
        x = F.relu(x)

        return x.view(-1)


class CNNDecoder(nn.Module):
    def __init__(self, in_channels, code_size):
        super(CNNDecoder, self).__init__()

        self.fc = nn.Linear(code_size, 16 * 3 * 3)