# region snntorch
import torch
import torch.nn as nn
import snntorch as snn


class customSNet(nn.Module):
    """
    simple SCNN with LIF neurons
    """

    def __init__(self, num_steps, beta, threshold=1.0, spike_grad=snn.surrogate.fast_sigmoid(slope=25), num_class=35):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)  # Batch Norm after conv1
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)  # Batch Norm after conv2
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        self.fc1 = nn.Linear(448, 128)  # 448 for rbf activations or spikes
        self.bn3 = nn.BatchNorm1d(128)  # Batch Norm after fc1
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)  # Batch Norm after fc2
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        self.fc3 = nn.Linear(64, num_class)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

    def forward(self, x):
        batch_size_curr = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk5_rec = []
        mem5_rec = []

        for step in range(self.num_steps):
            cur1 = self.pool(self.bn1(self.conv1(x)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.bn3(self.fc1(spk2.view(batch_size_curr, -1)))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.bn4(self.fc2(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc3(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        return torch.stack(spk5_rec), torch.stack(mem5_rec)


# region spikingjelly
import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer


class SJNetwork(nn.Module):
    """
    Simple Spiking Neural Network with:
    2 Convolutional(1d) layers,
    2 Linear layers,
    1 voting layer for classification. (average pooling over the last dimension)
    """

    def __init__(self, channels=128, output_size=10, spiking_neuron: callable = None, **kwargs):
        """
        Args:
            channels (int): number of channels in the convolutional layers.
            output_size (int): number of classes in the classification task.
            spiking_neuron (callable): the spiking neuron model.
            **kwargs: the parameters of the spiking neuron model.
        """
        super().__init__()
        voting_pool = 4
        size_last_linear = output_size * voting_pool
        conv = [
            layer.Conv1d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm1d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool1d(2),
            layer.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm1d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool1d(2),
        ]
        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            # layer.Linear(channels * 4 * 4, 512),
            layer.Linear(channels * 4, 128),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(128, size_last_linear),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer(voting_pool)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_fc(x)
        return x
