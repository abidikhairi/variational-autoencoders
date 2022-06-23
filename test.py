import argparse
import json
import torch as th
from torchvision import datasets, transforms

from models import VanillaAE
from utils import viz_reconstructed_images


def main(args):
    model_type = args.model_type
    dataset = args.dataset
    model_path = args.model_path
    data_path = args.data_path
    model_init_path = args.model_init_path

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1) if model_type == 'vanilla-ae' else x),
    ])

    if dataset == 'mnist':
        testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    
    testloader = th.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

    with open(model_init_path, 'r') as f:
        model_init = json.load(f)
    
    if model_type == 'vanilla-ae':
        model = VanillaAE(**model_init)
    else:
        raise NotImplemented('model {} type not implemented'.format(model_type))

    model.load_state_dict(th.load(model_path))
    model.eval()

    with th.no_grad():
        images, _ = next(iter(testloader))
        _, x_rec = model(images)
        x_rec = x_rec.view(-1, 28, 28).numpy()
        images = images.view(-1, 28, 28).numpy()
        viz_reconstructed_images(images, x_rec, n_images=images.shape[0], save_path='./figures/test-{}-{}.png'.format(model_type, dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-type', type=str, default='vanilla-ae', choices=['vanilla-ae', 'cnn-ae'])
    parser.add_argument('--model-path', type=str, default='./results/vanilla-ae.pt')
    parser.add_argument('--model-init-path', type=str, default='./results/vanilla_ae_init.json')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist'])
    parser.add_argument('--data-path', type=str, default='./data/')

    args = parser.parse_args()
    main(args)

