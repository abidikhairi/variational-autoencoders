import argparse
import json
import torch as th

from models import VanillaAE, CNNAE
from utils import viz_images


def main(args):
    model_type = args.model_type
    model_path = args.model_path
    model_init_path = args.model_init_path
    n_images = args.n_images
    
    with open(model_init_path, 'r') as f:
        model_init = json.load(f)
    
    if model_type == 'vanilla-ae':
        model = VanillaAE(**model_init)
    elif model_type == 'cnn-ae':
        model = CNNAE(**model_init)
    else:
        raise NotImplemented('model {} type not implemented'.format(model_type))

    model.load_state_dict(th.load(model_path))
    model.eval()

    with th.no_grad():
        # sample from the latent space (gaussian)
        noise = th.randn(n_images, model.code_size)
        images = model.decoder(noise)

        images = images.view(n_images, 28, 28).numpy()
        viz_images(images, n_images, save_path='./figures/test-generated-images-{}.png'.format(model_type))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-type', type=str, default='vanilla-ae', choices=['vanilla-ae', 'cnn-ae'])
    parser.add_argument('--model-path', type=str, default='./results/vanilla-ae.pt')
    parser.add_argument('--model-init-path', type=str, default='./results/vanilla_ae_init.json')
    parser.add_argument('--n-images', type=int, default=10, help='number of images to generate')

    args = parser.parse_args()
    main(args)

