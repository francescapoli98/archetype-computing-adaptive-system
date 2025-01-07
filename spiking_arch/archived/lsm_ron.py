from typing import (
    List,
    Literal,
    Tuple,
    Union,
)
import torch
from torch import nn

from acds.archetypes.utils import (
    get_hidden_topology,
    spectral_norm_scaling,
)

class LiquidRON(nn.Module):

    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        rho: float,
        input_scaling: float,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cpu",
        win_e=2,
        win_i=1,
        w_e=0.5,
        w_i=0.2,
        reg=None
    ):
        """Initialize the RON model."""
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        
        self.win_e = win_e
        self.win_i = win_i
        self.w_e = w_e
        self.w_i = w_i
        self.Ne = 800     # Number of excitatory neurons
        self.Ni = 200     # Number of inhibitory neurons
        self.reg = reg
        
        # Initialize weights
        self.readout = torch.rand(self.Ne + self.Ni, dtype=torch.float32)
        self.re = torch.rand(self.Ne, dtype=torch.float32)
        self.ri = torch.rand(self.Ni, dtype=torch.float32)
        self.a = torch.cat((0.02 * torch.ones(self.Ne), 0.02 + 0.08 * self.ri))
        self.b = torch.cat((0.2 * torch.ones(self.Ne), 0.25 - 0.05 * self.ri))
        self.c = torch.cat((-65 + 15 * self.re**2, -65 * torch.ones(self.Ni)))
        self.d = torch.cat((8 - 6 * self.re**2, 2 * torch.ones(self.Ni)))
        self.v = -65 * torch.ones(self.Ne + self.Ni, dtype=torch.float32)  # Initial values of v #-65
        self.u = self.v * self.b
        self.U = torch.cat((self.win_e * torch.ones(self.Ne), self.win_i * torch.ones(self.Ni)))
        
        self.S = torch.cat((
            self.w_e * torch.rand(self.Ne + self.Ni, self.Ne, dtype=torch.float32), 
            -self.w_i * torch.rand(self.Ne + self.Ni, self.Ni, dtype=torch.float32)
        ), dim=1)

        h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid, dtype=torch.float32) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid, dtype=torch.float32) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)
        
    def forward(self, data):
        print('v shape: ', self.v.size(), '\n u shape:', self.u.size(), '\n data size: ', data.size())
        ##, self.v.clone().unsqueeze()), dim=1
        v = torch.zeros(data.size(0), self.n_hid).to(self.device) ## voltage (membrane potential)
        u = torch.zeros(data.size(0), self.n_hid).to(self.device) ## recovery variable
        # you should update v and u starting from their self. value
        spikes = []  # spike timings
        states = []  # here we construct the matrix of reservoir states
    
        for t in range(len(data)):  # simulation of 1000 ms
            I = data[t] * self.U  # Input current (scaled)
            spike = (v >= 30) * 1.0
            # spike = (v >= 30).nonzero(as_tuple=False).squeeze()
            # spike = spike.long() 
            spikes.append(spike)
            
            # print("Indices of spike neurons:", len(spike))
            # spikes.append(torch.stack((t + torch.zeros_like(spike), spike), dim=0))
            # if spike.numel() > 0:  # Check if any neurons have spike
            # print('v[spike]: ', v[0][spike], ' shape: ', v[0][spike].size(), '\n c[spike]: ', self.c[spike], ' shape: ', self.c[spike].size())
            v[spike] = self.c[spike]  # Reset voltage of neurons that spike
            u[spike] = u[spike] + self.d[spike]  # Update recovery variable for neurons that spike

            I = I + torch.sum(self.S[:, spike], dim=1)  # Add synaptic input
            
            v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)  # Voltage update (Euler method)
            u = u + self.a * (self.b * v - u)  # Membrane recovery variable update
        
            # Append the spike states (voltage >= 30)
            states.append(v >= 30)
            spikes = torch.cat(spikes, dim=0)  # Concatenate spike timings

        # Return the states, voltage, recovery variable, and spike timings
        # print('states shape: ', torch.stack(states, dim=0).size(), 'v shape: ', v.size(), 'u size: ', u.size, 'spikes: ', spikes.size())
        return torch.stack(states, dim=0), v, u, spikes
    
    def train(self, data, target):
        states, v, u, spikes = self.forward(data)
        
        # Handle regularization (if specified)
        if self.reg is not None:
            self.readout = torch.linalg.pinv(states.T @ states + torch.eye(states.shape[1], dtype=torch.float32) * self.reg) @ states.T @ target
        else:
            self.readout = torch.linalg.pinv(states) @ target
            
        return states @ self.readout, v, u, spikes
