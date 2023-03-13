""" 
Code to generate biased Synbols dataset from Docker container.



"""


# Some imports and utils.

import synbols
from synbols.data_io import pack_dataset
from synbols import generate
from synbols import drawing

from synbols.visualization import plot_dataset
from pprint import pprint
import matplotlib.pyplot as plt
%matplotlib inline
def show_ds(attr_sampler, verbose=False, show_mask=False):
    """Generate and show a Synbols dataset from an attribute sampler."""
    x, mask, y = pack_dataset(generate.dataset_generator(attr_sampler, 100, generate.flatten_mask))
    if verbose:
        print("Example of a label")
        pprint(y[0])

    plt.figure('dataset', figsize=[5, 5])
    plot_dataset(x, y,h_axis=None,v_axis=None, n_row=10, n_col=10)

    plt.show()
    if show_mask:
        plt.figure('dataset', figsize=[5, 5])
        plot_dataset(mask, y,h_axis=None,v_axis=None, n_row=10, n_col=10)

        plt.show()


bg = drawing.MultiGradient(alpha=0.5, n_gradients=2, types=('linear', 'radial'))
fg = drawing.ImagePattern(root='/images')

im_bg = drawing.ImagePattern(root='/images')

letters = ["s", "t"]
samplers = []
im_samplers = []

for l in letters:
    samplers.append(generate.basic_attribute_sampler(
        resolution=(224, 224),
        foreground=fg,
        background=bg,
        inverse_color=False,
        char=l
    ))
    im_samplers.append(generate.basic_attribute_sampler(
        resolution=(224, 224),
        foreground=fg,
        background=im_bg,
        inverse_color=False,
        char=l
    ))

    
    
# show_ds(attr_sampler)

SAVE_DIR = ""

letters_to_id = {
    "s": 0,
    "t": 1,
}

import numpy as np
def save_dataset(name, letter, sampler, num_samples=10):
    x, mask, _ = pack_dataset(generate.dataset_generator(sampler, num_samples, generate.flatten_mask))
    
    y = np.array([letters_to_id[letter]]*num_samples)
    np.save(f"{name}_x.npy", x)
    np.save(f"{name}_mask.npy", mask)
    np.save(f"{name}_y.npy", y)


DATASET_SIZE = 1000

for i in range(2):
    save_dataset(f"{letters[i]}_grad", letters[i], samplers[i], num_samples=DATASET_SIZE)
    save_dataset(f"{letters[i]}_im", letters[i], im_samplers[i], num_samples=DATASET_SIZE)