from typing import (
    List,
    Literal,
    Tuple,
    Union,
)

import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate

import numpy as np
import random

from acds.archetypes.utils import (
    get_hidden_topology,
    spectral_norm_scaling,
)

from spiking_arch.snn_utils import *



class MixedRON(nn.Module):

    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        rho: float,
        input_scaling: float,
        threshold: float,
        resistance: float,
        capacitance: float,
        reset: float,
        p:float,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cpu"
    ):
        """Initialize the RON model.

        Args:
            n_inp (int): Number of input units.
            n_hid (int): Number of hidden units.
            dt (float): Time step.
            gamma (float or tuple): Damping factor. If tuple, the damping factor is
                randomly sampled from a uniform distribution between the two values.
            epsilon (float or tuple): Stiffness factor. If tuple, the stiffness factor
                is randomly sampled from a uniform distribution between the two values.
            rho (float): Spectral radius of the hidden-to-hidden weight matrix.
            input_scaling (float): Scaling factor for the input-to-hidden weight matrix.
                Wrt original paper here we initialize input-hidden in (0, 1) instead of (-2, 2).
                Therefore, when taking input_scaling from original paper, we recommend to multiply it by 2.
            topology (str): Topology of the hidden-to-hidden weight matrix. Options are
                'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz'. Default is
                'full'.
            reservoir_scaler (float): Scaling factor for the hidden-to-hidden weight
                matrix.
            sparsity (float): Sparsity of the hidden-to-hidden weight matrix.
            device (str): Device to run the model on. Options are 'cpu' and 'cuda'.
        """
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = (
                torch.rand(n_hid, requires_grad=False, device=device)
                * (gamma_max - gamma_min)
                + gamma_min
            )
        else:
            self.gamma = gamma
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = (
                torch.rand(n_hid, requires_grad=False, device=device)
                * (eps_max - eps_min)
                + eps_min
            )
        else:
            self.epsilon = epsilon
            
        #### TO BE DIVIDED IN SPIKING AND HARMONIC
        self.p = p ## FINE TUNE
        self.portion = n_hid*p
        # h2h_h = get_hidden_topology(harmonic, topology, sparsity, reservoir_scaler)
        # self.h2h_h = spectral_norm_scaling(h2h_h, rho)
        # self.h2h_h = nn.Parameter(h2h_h, requires_grad=False)
        # self.spiking = n_hid-harmonic
        # h2h_s = get_hidden_topology(spiking, topology, sparsity, reservoir_scaler)
        # self.h2h_s = spectral_norm_scaling(h2h_s, rho)
        # self.h2h_s = nn.Parameter(h2h_s, requires_grad=False)
        # Combine h2h_h and h2h_s into one parameter
        # combined_h2h = torch.cat((self.h2h_h, self.h2h_s), dim=0)  # Adjust dim based on your desired concatenation axis
        # self.h2h = nn.Parameter(combined_h2h, requires_grad=False)
        h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)
        
        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)

        ## FINE TUNE
        self.threshold = threshold 
        self.R = resistance
        self.C = capacitance
        self.reset = reset # initial membrane potential
    
        
## what we could do is, create a second cell
    def activation_layer(
        self, x: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and its derivative.

        Args:
            x (torch.Tensor): Input tensor.
            hy (torch.Tensor): Current hidden state.
        """        
        f = torch.tanh(torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias)
        harmonic_hiddens = f[:, :, :self.portion, :]
        spiking_hiddens = f[:, :, self.portion:, :]
        hy_harmonic, hz = harmonic_osc(harmonic_hiddens, hy, hz)
        hy_spiking, spikes = spiking_osc(spiking_hiddens)
        return hy_harmonic, hz, hy_spiking, spikes
    
    def spiking_osc(self, act, u):
        spike = (u > self.threshold) * 1.0
        # hy was previously weighted with self.w and x was weighted with R --> now I use reservoir weight
        u[spike == 1] = self.reset  # Hard reset only for spikes
        # tau = R * C
        u = u + ((self.R*self.C)*self.dt)*(-u + act)
        return spike, u
        # u -= spike*self.threshold # soft reset the membrane potential after spike
        ## plot membrane potential with thresholds and positive spikes
    
    def harmonic_osc(self, act, hy, hz):
        hz = hz + self.dt * act - self.gamma * hy - self.epsilon * hz

        hy = hy + self.dt * hz
        return hy, hz

       
    def forward(self, x: torch.Tensor):
        """Forward pass on a given input time-series.

        Args:
            x (torch.Tensor): Input time-series shaped as (batch, time, input_dim).

        Returns:
            torch.Tensor: Hidden states of the network shaped as (batch, time, n_hid).
            list: List containing the last hidden state of the network.
        """
        hy_m_list = [] # hy from harmonic mechanical oscillators
        hz_list = [] #hz from mechanical oscillators too
        hy_u_list = [] #membrane potential = hy from spiking oscillators
        spike_list = [] #spikes
        
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device) #x.size(0)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)
        # u = torch.zeros(x.size(0), self.n_hid).to(self.device)
        
        f = self.activation_layer(x, hy, hz)
        
        
        # Combine h2h_h and h2h_s into one parameter
        comb_h2h = torch.cat((self.h2h_h, self.h2h_s), dim=0)  # Adjust dim based on your desired concatenation axis
        # self.h2h = nn.Parameter(combined_h2h, requires_grad=False)
        for t in range(x.size(1)):      
                hy_m, hz, hy_u, spk = self.activation_layer(x[:, t], hy, hz)
                hy_m_list.append(hy_m)
                hz_list.append(hz)
                hy_u_list.append(hy_u)
                spike_list.append(spk)
        return hy_m_list, hz_list, hy_u_list, spike_list 