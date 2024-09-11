import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import utils as sutils
from snntorch.functional import quant


class customSNet_0(nn.Module):
    def __init__(self, num_steps, beta, threshold=1.0, spike_grad=snn.surrogate.fast_sigmoid(slope=25), num_class=10):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc1 = nn.Linear(448, 128)
        # self.fc1 = nn.Linear(896, 128)  # use 6720 for real spektogram, 896 for spikes or rbf activity
        # self.fc1 = nn.Linear(6720, 128) # use 6720 for real spektogram, 896 for spikes or rbf activity
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc2 = nn.Linear(128, 64)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc3 = nn.Linear(64, num_class)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        batch_size_curr = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []

        for step in range(self.num_steps):
            # print('x0', x.shape)
            cur1 = self.pool(self.conv1(x))
            # cur1 = self.conv1(x)
            # print('x1', x.shape)
            spk1, mem1 = self.lif1(cur1, mem1)
            # print('x2', spk1.shape, mem1.shape)
            cur2 = self.pool(self.conv2(spk1))
            # cur2 = self.conv2(spk1)
            # print('x3', cur2.shape)
            spk2, mem2 = self.lif2(cur2, mem2)
            # print('x4', spk2.shape, mem2.shape)
            cur3 = self.fc1(spk2.view(batch_size_curr, -1))
            # print('x5', cur3.shape)
            spk3, mem3 = self.lif3(cur3, mem3)
            # print('x6', spk3.shape, mem3.shape)
            cur4 = self.fc2(spk3)
            # print('x7', cur4.shape)
            spk4, mem4 = self.lif4(cur4, mem4)
            # print('x8', spk4.shape, mem4.shape)
            cur5 = self.fc3(spk4)
            # print('x9', cur5.shape)
            spk5, mem5 = self.lif5(cur5, mem5)
            # print('x10', spk5.shape, mem5.shape)

            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        return torch.stack(spk5_rec), torch.stack(mem5_rec)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# updated version with batch norm and decaying learning rate

import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate
from torch.optim import lr_scheduler


class customSNet_1(nn.Module):
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
    def __init__(self, channels=128, output_size=10, spiking_neuron: callable = None, **kwargs):
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
