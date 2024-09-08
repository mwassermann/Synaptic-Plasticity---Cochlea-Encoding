import librosa
import numpy as np
import torch
from scipy.signal.windows import gaussian
from scipy.signal import ShortTimeFFT


# region tuning curves
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


# region rbf
class RBFNetwork:
    def __init__(self, input_dim, num_centers, sigma):
        self.centers = np.random.rand(num_centers, input_dim)  # Initialize centers randomly
        # print("centers: ", self.centers.shape)
        self.sigma = sigma

    def rbf(self, x):
        # print("x: ", x.shape)
        # print("centers: ", self.centers.shape)
        # return np.exp(-np.linalg.norm(x - self.centers, axis=1) ** 2 / (2 * self.sigma ** 2))
        def compute_distances(xi):
            # xi - self.centers creates a new array where each center is subtracted from xi
            # np.linalg.norm(..., axis=1) computes the norm along the axis of the centers
            return np.linalg.norm(xi - self.centers, axis=1)

        norms = np.apply_along_axis(compute_distances, 1, x)
        return np.exp(-(norms**2) / (2 * self.sigma**2))

    def predict(self, X):
        # print("## predict ##")
        # print("X: ", X.shape)
        y = self.rbf(X)
        # print("y: ", y.shape)
        # normalize to 0 - 1 along the batch dimension
        try:
            y = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))
        except:  # if the denominator is zero or close to zero
            y = 1
        return y


def rbf_encode_audio(audio, sample_rate, SFT, n_mels=128, num_rbf=16, sigma=1.0):
    # spec = spectrogram(audio, sample_rate, n_mels)
    # spec = np.abs(spec)
    # print('type of audio:', type(audio))
    # print('audio shape:', audio.shape)

    if type(audio) is torch.Tensor:
        audio = audio.numpy()

    spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=252)
    # print("shape spec: ", spec.shape)

    # mel_spec = SFT.spectrogram(audio)
    # print("shape mel_spec: ", mel_spec.shape)
    rbf_network = RBFNetwork(spec.shape[0], num_rbf, sigma)
    rbf_activations = rbf_network.predict(spec.T)  # transpose to get the batch dimension first
    return rbf_activations, spec


# poisson spikes from rbf activity
def encode_to_spikes_rbf(data, sr, tau_m=20.0, R=1.0, V_th=1.0, V_reset=0.0, SFT=None):
    batch_size = data.size(0)
    num_features = 16
    num_time_steps = 64
    spikes = torch.zeros(batch_size, num_features, num_time_steps, device=data.device)
    rbfs = []
    mel_specs = []

    for i in range(batch_size):
        rbf_activations, mel_spec = rbf_encode_audio(data[i], sr, SFT=SFT, num_rbf=16, n_mels=16)
        rbfs.append(rbf_activations)
        mel_specs.append(mel_spec)

        # print("rbf_activations: ", rbf_activations.shape)
        # print("mel_spec: ", mel_spec.shape)
        # print("min max rbf_activations: ", np.min(rbf_activations), np.max(rbf_activations))
        # print("min max mel_spec: ", np.min(mel_spec), np.max(mel_spec))

        spike_prob_scale = 1.7

        rbf_activations_traversed = rbf_activations.T
        spik_probs = (
            rbf_activations_traversed / np.max(rbf_activations_traversed, axis=1, keepdims=True) * spike_prob_scale
        )
        spike_trains = np.random.poisson(spik_probs[...], size=rbf_activations_traversed.shape)
        spike_trains = np.clip(spike_trains, 0, 1)
        spikes[i] = torch.from_numpy(spike_trains)

    return spikes


# region mel spectogram
def encode_to_spikes_melspec(data, sr, tau_m=20.0, R=1.0, V_th=1.0, V_reset=0.0, SFT=None):
    batch_size = data.size(0)
    num_features = 16
    num_time_steps = 64
    spikes = torch.zeros(batch_size, num_features, num_time_steps, device=data.device)
    rbfs = []
    mel_specs = []

    for i in range(batch_size):
        rbf_activations, mel_spec = rbf_encode_audio(data[i], sr, SFT=SFT, num_rbf=16, n_mels=16)
        rbfs.append(rbf_activations)
        mel_specs.append(mel_spec)

        # print("rbf_activations: ", rbf_activations.shape)
        # print("mel_spec: ", mel_spec.shape)
        # print("min max rbf_activations: ", np.min(rbf_activations), np.max(rbf_activations))
        # print("min max mel_spec: ", np.min(mel_spec), np.max(mel_spec))

        spike_prob_scale = 1.7

        # TODO get spikes from the mel_spectograms
        # first normalize mel_spec
        mel_spec_min = mel_spec.min()
        mel_spec_max = mel_spec.max()
        mel_spec_norm = (mel_spec - mel_spec_min) / (mel_spec_max - mel_spec_min)
        mel_spec_scaled = mel_spec_norm * spike_prob_scale

        # Generate spike trains based on scaled mel spectrogram
        # For simplicity, using Poisson distribution to model spike generation
        # Each value in mel_spec_scaled represents a spike rate
        mel_spec_tensor = torch.tensor(mel_spec_scaled, dtype=torch.float32)
        spike_trains = torch.poisson(mel_spec_tensor)

        # Clamp values to 0 or 1 because Poisson can generate values >1
        spike_trains = torch.clamp(spike_trains, max=1)

        # Assign generated spike trains to spikes tensor for the batch
        spikes[i] = spike_trains
    return spikes
