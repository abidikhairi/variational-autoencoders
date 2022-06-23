import argparse
import json



def main(args):
    model_type = args.model_type
    model_path = args.model_path
    data_path = args.data_path
    model_init_path = args.model_init_path






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-type', type=str, default='vanilla-ae', choices=['vanilla-ae', 'cnn-ae'])
    parser.add_argument('--model-path', type=str, default='./results/vanilla-ae.pt')
    parser.add_argument('--model-init-path', type=str, default='./results/vanilla_ae_init.json')
    parser.add_argument('--data-path', type=str, default='./data/mnist/')

    args = parser.parse_args()
    main(args)

