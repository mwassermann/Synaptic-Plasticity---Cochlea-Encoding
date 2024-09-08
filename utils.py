import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.ndimage import convolve1d
from scipy.signal.windows import exponential, gaussian
from scipy.signal import square, ShortTimeFFT
from scipy.signal import stft, istft
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import spikingjelly.clock_driven as cd
import spikingjelly.datasets as sjds

# from spikingjelly.clock_driven import neuron, encoding, functional, layer, surrogate
from spikingjelly.activation_based import neuron, layer, learning

import torchvision.transforms as transforms

import IPython.display as ipd
from tqdm.notebook import tqdm


def create_individual_kernel(kernel_size, tau, g_std, middle, assume_circular=False):
    # match the kernel size such that middle is in the middle of the kernel
    original_kernel_size = kernel_size
    shift = middle - kernel_size // 2
    if not assume_circular:
        # extend the kernel size to accommodate the shift
        kernel_size = kernel_size + 2 * np.abs(shift)

    k_gauss = gaussian(kernel_size, std=g_std, sym=True)
    k_exp = exponential(kernel_size, tau=tau, sym=True)
    combined = k_gauss + k_exp

    if assume_circular:
        combined = np.roll(combined, shift)
    else:
        # crop the kernel to the original size
        if shift < 0:  # if negative shift, take the end of the wrapper kernel
            combined = combined[-original_kernel_size:]
        else:  # if positive shift, take the beginning of the wrapper kernel
            combined = combined[:original_kernel_size]

    combined /= np.max(combined)  # Normalize the kernel
    return combined
