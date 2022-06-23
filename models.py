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
    def __init__(self, input_size, hidden_size, output_size, gray_scale=False):
        super(VanillaAE, self).__init__()
        
        self.gray_scale = gray_scale
        self.encoder = MLP(input_size, hidden_size, output_size)
        self.decoder = MLP(output_size, hidden_size, input_size)


    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        if self.gray_scale:
            x_rec = th.sigmoid(x_rec)
        return z, x_rec


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, code_size):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc = nn.Linear(16 * 3 * 3, code_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):
        batch_size = x.shape[0]

        x = self.pool(self.conv1(x))
        x = F.relu(x)
        x = self.conv3(self.conv2(x))
        x = self.pool(x)
        x = F.relu(x)

        return self.fc(x.view(batch_size, -1))


class CNNDecoder(nn.Module):
    def __init__(self, out_channels, code_size):
        super(CNNDecoder, self).__init__()

        self.fc = nn.Linear(code_size, 16 * 3 * 3)
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3)
        self.conv2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3)
        self.conv3 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=5)
        self.conv4 = nn.ConvTranspose2d(in_channels=2, out_channels=out_channels, kernel_size=4)


    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 3, 3)

        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=1.5, mode='nearest')
        x = F.interpolate(self.conv2(x), scale_factor=1.5, mode='nearest')
        x = F.interpolate(self.conv3(x), scale_factor=1.5, mode='nearest')
        x = F.relu(self.conv4(x))
        
        return x


class CNNAE(nn.Module):
    def __init__(self, in_channels, code_size):
        super(CNNAE, self).__init__()

        self.encoder = CNNEncoder(in_channels, code_size)
        self.decoder = CNNDecoder(in_channels, code_size)


    def forward(self, images):
        z = self.encoder(images)
        x_rec = self.decoder(z)
        return z, x_rec
