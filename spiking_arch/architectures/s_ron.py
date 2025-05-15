'''
BEST VALUES:
SMNIST
--threshold: 0.008 --reset: 0.004 --rc: 7.0

M-G
--rc 5 --threshold 0.009 --reset 0.004 --bias 0.1
'''
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

from acds.archetypes.utils import (
    get_hidden_topology,
    spectral_norm_scaling,
)

from spiking_arch.snn_utils import *



class SpikingRON(nn.Module):

    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        #grid search params
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        rho: float,
        input_scaling: float,
        threshold: float,
        # resistance: float,
        # capacitance: float,
        rc: float,
        reset: float,
        bias: float,
        #no grid search
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cpu",
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
            
        self.threshold = threshold
        # self.R = resistance
        # self.C = capacitance
        self.rc = rc #resistance + capacitance
        self.reset = reset # initial membrane potential 
        
        #### to be changed to spiking dynamics
        h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)
        
        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        
        self.bias = bias
        
## plot hy, hz, x, u and spikes to see the variation
    def cell(
        self, x: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and its derivative.

        Args:
            x (torch.Tensor): Input tensor ---> u(t) == I(t) external current
            hy (torch.Tensor): Current hidden state ----> position (y)
            hz (torch.Tensor): Current hidden state derivative ----> velocity (y'=z)
            u (torch.Tensor): Member potential 
        """
        spike, u = self.spiking_layer(x, hy, u) 
        
        hz = hz + self.dt * ( 
            u 
            - self.gamma * hy 
            - self.epsilon * hz
        )

        hy = hy + self.dt * hz
        return hy, hz, u, spike
    
    ##################### 
    '''
    from the snntorch tutorial:
    
        # LIF w/Reset mechanism
        
        def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
            tau_mem = R*C
            spk = (mem > threshold)
            mem = mem + (time_step/tau_mem)*(-mem + cur*R) - spk*threshold  # every time spk=1, subtract the threhsold
            return mem, spk
    '''
    ##################### 
    
    #### CHECK!!!
    ## I == W hy + V x where W and V need to be initialized at the beginning and are fixed (reservoir)
    ## u == member potential and is not correspondent to hy because it's an internal parameter to get spikes (don't forget to update!)
    #### output =  spikes! but you can add the weights to change from the 0-1 binary output
    '''
    returns:
    - spike = binary spikes (tensor)
    - u = membrane potential (tensor)
    '''
    def spiking_layer(self, x, hy, u):
        #### ADD THE PRINTS HERE TO A OUTPUT FILE
        spike = (u > self.threshold) * 1.0
        # hy was previously weighted with self.w and x was weighted with R --> now I use reservoir weights
  
        u[spike == 1] = self.reset  # Hard reset only for spikes

        ## for NMNIST dataset, deal with x2h param to deal with a timestep dim of 20!
            ## x: 1 x n   x2h: n x hid
        # print('ACTIVATION\nx size: ', x.size(), '\nx2h size: ', self.x2h.size()) ## make sure x and x2h is the same for RON and baseline
        u_dot = - u + (torch.matmul(hy, self.h2h) + torch.matmul(x, self.x2h) + self.bias) # u dot (update) 
        u = u + (u_dot * self.rc # tau = self.rc = (self.R*self.C)
                 )*self.dt # multiply to tau and dt
        
        return spike, u
        # u -= spike*self.threshold # soft reset the membrane potential after spike
        # OLD CODE: u += ((- self.w * hy) + (self.R*x))*(self.R*self.C) - spike*self.threshold  
        
    def forward(self, x: torch.Tensor):
        """Forward pass on a given input time-series.

        Args:
            x (torch.Tensor): Input time-series shaped as (batch, time, input_dim).

        Returns:
            torch.Tensor: Hidden states of the network shaped as (batch, time, n_hid).
            list: List containing the last hidden state of the network.
        """
        hy_list, hz_list, u_list, spike_list = [], [], [], []
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device) #x.size(0)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)
        u = torch.zeros(x.size(0), self.n_hid).to(self.device)
        # print('input x size: ', x.size(), '\ntimestep: ', x[:, 1].size())
        for t in range(x.size(1)):
            hy, hz, u, spk = self.cell(x[:, t], hy, hz, u)
            hy_list.append(hy)
            hz_list.append(hz)
            u_list.append(u)
            spike_list.append(spk)
        return hy_list, hz_list, u_list, spike_list