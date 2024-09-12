import torch
import numpy as np
import librosa
from spikingjelly.activation_based import neuron, functional
from scipy.signal.windows import gaussian
from scipy.signal import ShortTimeFFT
from scipy.special import softmax


# region spectogram
def convert_audio_to_spectrogram(
    waves,
    sr=16000,
    n_mels=20,
    n_fft=2048,
    hop_length=252,
):
    # make sure waves is a numpy array
    if isinstance(waves, torch.Tensor):
        waves = waves.numpy()
    elif not isinstance(waves, np.ndarray):
        waves = np.array(waves)

    # print("waves shape: ", waves.shape)

    # # using ShortTimeFFT
    # g_std = 10  # standard deviation for Gaussian window in samples
    # win_size = 40  # window size in samples
    # win_gauss = gaussian(win_size, std=g_std, sym=True)  # symmetric Gaussian wind.
    # STFT = ShortTimeFFT(win_gauss, hop=2, fs=sr, mfft=40, scale_to="psd")
    # specs = STFT.spectrogram(waves)
    # spec_frequencies = STFT.f

    # using mel spectrogram from librosa
    fmin = 0
    fmax = sr // 2
    num_time_steps = (sr // hop_length) + 1
    # print("num time steps: ", num_time_steps)
    specs = librosa.feature.melspectrogram(y=waves, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spec_frequencies = librosa.mel_frequencies(n_mels, fmin=fmin, fmax=fmax)

    return specs, spec_frequencies


# region spike train
# encoding to spikes with Integrate-and-Fire (IF) neurons
def generate_spike_trains_IF(inputs, tau_m=0.3, V_th=1.0, V_reset=0.0, leaky=False):
    batch_size, num_freqs, timesteps = inputs.shape

    # Normalizing spectrogram
    inputs = torch.from_numpy(inputs)
    # normalize each spectrogram along the batch dimension
    # Normalize each spectrogram along the batch dimension (dim 1 and dim 2)
    inputs = (inputs - inputs.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / (
        inputs.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        - inputs.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    )

    if leaky:
        # Create leaky IF neurons
        lif_neurons = neuron.LIFNode(v_threshold=V_th, v_reset=V_reset)
    else:
        # Create IF neurons
        lif_neurons = neuron.IFNode(v_threshold=V_th, v_reset=V_reset)
    functional.set_step_mode(lif_neurons, "m")
    # Simulating the neuron response
    spike_trains = []
    for t in range(timesteps):
        # Apply input to neurons
        input_current = inputs[:, :, t] / tau_m  # Scale input as per membrane time constant
        spikes = lif_neurons(input_current)
        spike_trains.append(spikes)

    # Resetting the neuron states after processing
    functional.reset_net(lif_neurons)

    # Stacking to form the batch x num_freqs x timesteps tensor
    spike_trains = torch.stack(spike_trains, dim=2)

    return spike_trains.numpy()


# encoding to spikes with Poisson neurons
def generate_spike_trains_poisson(spike_probs):
    spike_probs = spike_probs
    max_pois_val = 100
    spike_probs = np.clip(spike_probs, 0, max_pois_val)
    spike_trains = np.random.poisson(spike_probs)
    spike_trains = np.clip(spike_trains, 0, 1)

    return spike_trains


# region tuning curves
def apply_TCs_kernels(specs, spec_frequencies, kernels):
    batch_size, num_freqs, timesteps = specs.shape
    # normalize each spectrogram along the batch dimension
    specs = (specs - np.min(specs, axis=(1, 2), keepdims=True)) / (
        np.max(specs, axis=(1, 2), keepdims=True) - np.min(specs, axis=(1, 2), keepdims=True)
    )

    filtered_specs = np.zeros((batch_size, len(kernels), timesteps))
    for i in range(batch_size):
        for idx, k in enumerate(kernels):
            k = np.interp(spec_frequencies, np.arange(len(k)), k)
            # filter the spectrogram with the kernel by multiplying element-wise along the time axis
            spike_probs = np.apply_along_axis(lambda x: x * k, 0, specs[i])
            spike_probs = np.average(spike_probs, weights=k, axis=0)
            filtered_specs[i, idx, ...] = spike_probs

    return filtered_specs


# convenience function with default parameters
def encode_to_spikes_TCs(waves, kernels, sr=16000, spike_encoding="IF", spike_prob_scale=1.0):
    specs, spec_frequencies = convert_audio_to_spectrogram(waves, sr, n_mels=40, hop_length=126)
    spike_probs = apply_TCs_kernels(specs, spec_frequencies, kernels)
    # normalize spike_probs to 0 - 1
    spike_probs = (spike_probs - np.min(spike_probs)) / (np.max(spike_probs) - np.min(spike_probs))

    if spike_encoding == "IF":
        spike_trains = generate_spike_trains_IF(spike_probs * spike_prob_scale)
    elif spike_encoding == "poisson":
        spike_trains = generate_spike_trains_poisson(spike_probs * spike_prob_scale)
    else:
        raise ValueError("Invalid spike encoding method")
    return spike_trains, spike_probs, specs, spec_frequencies


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
        # softmax normalization
        y = softmax(y, axis=1)

        return y


def rbf_encode_specs(specs, num_rbf=16, sigma=1.0):
    rbf_activations = np.zeros((specs.shape[0], num_rbf, specs.shape[-1]))
    for idx, spec in enumerate(specs):
        rbf_network = RBFNetwork(spec.shape[0], num_rbf, sigma)
        rbf_activation = rbf_network.predict(spec.T)  # transpose to get the batch dimension first
        rbf_activations[idx, ...] = rbf_activation.T
    return rbf_activations


# convenience function with default parameters
def encode_to_spikes_rbf(data, sr, num_rbf=16, sigma=1.0, spike_encoding="IF", spike_prob_scale=1.0):
    specs, freqs = convert_audio_to_spectrogram(data, sr, n_mels=40, hop_length=126)
    rbf_activations = rbf_encode_specs(specs, num_rbf, sigma)
    # normalize spike_probs to 0 - 1
    spike_probs = (rbf_activations - np.min(rbf_activations)) / (np.max(rbf_activations) - np.min(rbf_activations))
    if spike_encoding == "IF":
        spike_trains = generate_spike_trains_IF(spike_probs * spike_prob_scale, tau_m=0.5)
    elif spike_encoding == "poisson":
        spike_trains = generate_spike_trains_poisson(spike_probs * spike_prob_scale)
    else:
        raise ValueError("Invalid spike encoding method")

    return spike_trains, spike_probs, rbf_activations, specs, freqs
