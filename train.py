import logging
import argparse
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms

from models import VanillaAE


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def main(args):

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay

    device = th.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    trainset, validset = random_split(trainset, [55000, 5000])

    trainloader = th.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = th.utils.data.DataLoader(validset, batch_size=test_batch_size, shuffle=True)
    testloader = th.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    feature_size = 784
    hidden_size = 128
    code_size = 32

    model = VanillaAE(feature_size, hidden_size, code_size)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    for epoch in range(epochs):
        train_loss = []
        valid_loss = []

        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, x_rec = model(x)
            loss = loss_fn(x_rec, x)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        for x, _ in validloader:
            x = x.to(device)
            z, x_rec = model(x)
            loss = loss_fn(x_rec, x)

            valid_loss.append(loss.item())
        
        logger.info(f'epoch {epoch + 1} | train loss: {np.mean(train_loss):.4f} | valid loss: {np.mean(valid_loss):.4f}')

    test_loss = []
    for x, _ in testloader:
        x = x.to(device)
        z, x_rec = model(x)
        loss = loss_fn(x_rec, x)
        test_loss.append(loss.item())

    logger.info(f'test loss: {np.mean(test_loss):.4f}')
    
    with open('./results/vanilla-ae.pt', 'wb') as f:
        th.save(model.state_dict(), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Autoencoder Example on MNIST')
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=10e-6,)
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')

    args = parser.parse_args()
    
    main(args)