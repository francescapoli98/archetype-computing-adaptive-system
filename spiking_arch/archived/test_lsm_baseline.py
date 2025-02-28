import torch
from torch import nn
import snntorch as snn
import numpy as np
from acds.archetypes.utils import spectral_norm_scaling


class LiquidRON(nn.Module):
    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        dt: float,
        gamma: float,
        epsilon: float,
        rho: float,
        input_scaling: float,
        threshold: float,
        rc: float,
        reset: float,
        bias: float,
        win_e: int,
        win_i: int,
        w_e: float,
        w_i: float,
        Ne: int,
        Ni: int,
        topology="full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cuda",
    ):
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        self.threshold = threshold
        self.rc = rc
        self.reset = reset
        self.bias = bias

        # Reservoir connections
        h2h = np.concatenate(
            (w_e * np.random.rand(Ne + Ni, Ne), -w_i * np.random.rand(Ne + Ni, Ni)),
            axis=1,
        )
        h2h = torch.tensor(h2h, dtype=torch.float32, device=self.device)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=True)

        # Input connections
        input_scaling_values = np.concatenate((win_e * np.ones(Ne), win_i * np.ones(Ni)))
        x2h = torch.rand(n_inp, n_hid, dtype=torch.float32, device=self.device) * torch.tensor(
            input_scaling_values, device=self.device
        )
        self.x2h = nn.Parameter(x2h, requires_grad=True)

        # **Spiking Neuron Layer**
        self.snn_layer = snn.Leaky(
            beta=1 - (dt / rc),  # Decay factor based on time constant
            threshold=threshold,
            reset_mechanism="zero",
            learn_threshold=False,  # Keep threshold fixed
        )

    def forward(self, x: torch.Tensor):
        u_list, spike_list = [], []
        u = torch.zeros(x.size(0), self.n_hid).to(self.device)

        for t in range(x.size(1)):
            input_current = torch.matmul(x[:, t], self.x2h) + torch.matmul(u, self.h2h) + self.bias
            spk, u = self.snn_layer(input_current, u)  # Use snn.Leaky for membrane updates
            u_list.append(u)
            spike_list.append(spk)

        return torch.stack(u_list, dim=1), torch.stack(spike_list, dim=1)