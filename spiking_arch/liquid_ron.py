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
        self.v = -50 * torch.ones(self.Ne + self.Ni, dtype=torch.float32)  # Initial values of v #-65
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
        u, v = self.u.clone(), self.v.clone()  # Avoid in-place modification
        firings = []  # spike timings
        states = []  # here we construct the matrix of reservoir states
        
        for t in range(len(data)):  # simulation of 1000 ms
            I = data[t] * self.U  # Input current (scaled)
            fired = torch.where(v >= 30)[0]  # Indices of spikes
            firings.append(torch.stack((t + torch.zeros_like(fired), fired), dim=1))
            # if fired.numel() > 0:  # Check if any neurons have fired
            v[fired] = self.c[fired]  # Reset voltage of neurons that fired
            u[fired] = u[fired] + self.d[fired]  # Update recovery variable for neurons that fired
            print("v before firing:", v)
            print("Indices of fired neurons:", fired)

            v[fired] = self.c[fired]  # Reset voltage of neurons that fired
            u[fired] = u[fired] + self.d[fired]  # Reset u after firing
            I = I + torch.sum(self.S[:, fired], dim=1)  # Add synaptic input
            
            v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)  # Voltage update (Euler method)
            u = u + self.a * (self.b * v - u)  # Membrane recovery variable update
            
            # Append the spike states (voltage >= 30)
            states.append(v >= 30)

        firings = torch.cat(firings, dim=0)  # Concatenate spike timings

        # Return the states, voltage, recovery variable, and spike timings
        return torch.stack(states, dim=0), v, u, firings
    
    def train(self, data, target):
        states, v, u, firings = self.forward(data)
        
        # Handle regularization (if specified)
        if self.reg is not None:
            self.readout = torch.linalg.pinv(states.T @ states + torch.eye(states.shape[1], dtype=torch.float32) * self.reg) @ states.T @ target
        else:
            self.readout = torch.linalg.pinv(states) @ target
            
        return states @ self.readout, v, u, firings
