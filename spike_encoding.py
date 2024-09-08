import librosa
import numpy as np
import torch
from scipy.signal.windows import gaussian
from scipy.signal import ShortTimeFFT


def convert_audio_to_spectrogram(waves, sr=16000, spec_method="mel"):

    # make sure waves is a numpy array
    if isinstance(waves, torch.Tensor):
        waves = waves.numpy()
    elif not isinstance(waves, np.ndarray):
        waves = np.array(waves)

    # print("waves shape: ", waves.shape)
    if spec_method == "ShortTimeFFT":
        # using ShortTimeFFT
        g_std = 10  # standard deviation for Gaussian window in samples
        win_size = 40  # window size in samples
        win_gauss = gaussian(win_size, std=g_std, sym=True)  # symmetric Gaussian wind.
        STFT = ShortTimeFFT(win_gauss, hop=2, fs=sr, mfft=40, scale_to="psd")
        specs = STFT.spectrogram(waves)
        spec_frequencies = STFT.f

    elif spec_method == "mel":
        # using mel spectrogram from librosa
        # sr = 16000
        n_mels = 16
        n_fft = 2048
        hop_length = 252
        num_time_steps = (sr // hop_length) + 1
        # print("num time steps: ", num_time_steps)
        specs = librosa.feature.melspectrogram(y=waves, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        spec_frequencies = librosa.mel_frequencies(n_mels, fmin=0, fmax=sr / 2)

    else:
        raise ValueError("Invalid spectrogram method. Choose from 'ShortTimeFFT' or 'mel'")

    # print("specs shape: ", specs.shape)
    # print("spec frequencies shape: ", spec_frequencies.shape)
    # print("spec frequencies", spec_frequencies)

    return specs, spec_frequencies


def convert_spectogram_to_spike_trains(specs, spec_frequencies, kernels, spike_prob_scale=1.0):
    batch_size, num_freqs, timesteps = specs.shape
    # normalize each spectrogram along the batch dimension
    specs = (specs - np.min(specs, axis=(1, 2), keepdims=True)) / (
        np.max(specs, axis=(1, 2), keepdims=True) - np.min(specs, axis=(1, 2), keepdims=True)
    )
    spike_trains = np.zeros((batch_size, len(kernels), timesteps))

    for i in range(batch_size):
        for idx, k in enumerate(kernels):
            k = np.interp(spec_frequencies, np.arange(len(k)), k)
            scaled_kernel = k * spike_prob_scale
            # filter the spectrogram with the kernel by multiplying element-wise along the time axis
            spike_probs = np.apply_along_axis(lambda x: x * scaled_kernel, 0, specs[i])

            spike_probs = np.average(spike_probs, weights=k, axis=0)

            spike_train = np.random.poisson(spike_probs)
            spike_train = np.clip(spike_train, 0, 1)
            spike_trains[i, idx, ...] = spike_train

    return spike_trains


def encode_to_spikes_TCs(waves, kernels, sr=16000, spec_method="mel", spike_prob_scale=1.0):
    specs, spec_frequencies = convert_audio_to_spectrogram(waves, sr, spec_method)
    spike_trains = convert_spectogram_to_spike_trains(specs, spec_frequencies, kernels, spike_prob_scale)
    return spike_trains, specs, spec_frequencies
