import json
import numpy as np
import matplotlib.pyplot as plt


def save_init_params(params, save_path):
    """
    Save initial parameters.
    """
    if not isinstance(params, dict):
        raise TypeError('params must be a dict')
    
    with open(save_path, 'w') as f:
        json.dump(params, f)


def viz_reconstructed_images(original, reconstructed, n_images=10, save_path=None):
    """
    Visualize reconstructed images.
    """
    fig = plt.figure(constrained_layout=True, figsize=(20, 4))
    subfigs = fig.subfigures(nrows=2, ncols=1)

    images = np.concatenate([original, reconstructed], axis=0)
    
    for row, subfig in enumerate(subfigs):
        if row == 0:
            subfig.suptitle('Original')
        else:
            subfig.suptitle('Reconstructed')
        
        axs = subfig.subplots(nrows=1, ncols=n_images)

        for col, ax in enumerate(axs):            
            ax.imshow(images[row * n_images + col], cmap='gray')
            ax.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
