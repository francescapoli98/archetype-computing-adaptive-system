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
        capacity: float,
        reset: float,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cpu",
        p=float,
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
        self.harmonic = n_hid*p
        # h2h_h = get_hidden_topology(harmonic, topology, sparsity, reservoir_scaler)
        # self.h2h_h = spectral_norm_scaling(h2h_h, rho)
        # self.h2h_h = nn.Parameter(h2h_h, requires_grad=False)
        self.spiking = n_hid-harmonic
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
        self.C = capacity
        self.reset = reset # initial membrane potential
    
        
## what we could do is, create a second cell
    def cell(
        self, x: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and its derivative.

        Args:
            x (torch.Tensor): Input tensor.
            hy (torch.Tensor): Current hidden state.
            hz (torch.Tensor): Current hidden state derivative.
        """
        spike, u = spiking_osc(x, hy, u)
        
        hz = hz + self.dt * ( # should this change?
            torch.tanh(
                torch.matmul(x, self.x2h) + torch.matmul(hy, self.h2h) + self.bias
            )
            ### this is the part that technically describes the mechanical oscillators
            ## and that we'll now update with also spiking ones
            - self.gamma * hy
            - (self.epsilon * (hz*self.harmonic) + u*self.spiking)
        )

        hy = hy + self.dt * hz
        return hy, hz, u, spike

    #### CHECK!!!
    ## I == W hy + V x where W and V need to be initialized at the beginning and are fixed (reservoir)
    ## u == member potential and is not correspondent to hy because it's an internal parameter to get spikes (don't forget to update!)
    #### output =  spikes! but you can add the weights to change from the 0-1 binary output
    '''
    returns:
    - spike = binary spikes (tensor)
    - u = membrane potential (tensor)
    '''
    def spiking_osc(self, x, hy, u):
        spike = (u > self.threshold) * 1.0
        # hy was previously weighted with self.w and x was weighted with R --> now I use reservoir weight
        u[spike == 1] = self.reset  # Hard reset only for spikes
        # tau = R * C
        # u_dot = - u + (torch.matmul(hy, self.h2h) + torch.matmul(x, self.x2h)) # u dot (update) 
        # u = u + (u_dot * (self.R*self.C))*self.dt # multiply to tau and dt
        u = u + ((self.R*self.C)*self.dt)*(-u)
        return spike, u
        # u -= spike*self.threshold # soft reset the membrane potential after spike
        ## plot membrane potential with thresholds and positive spikes
    
    # def plot_u(self, u_list: List[torch.Tensor], resultroot):
    #     """Plot the membrane potential u over time for each neuron."""
    #     u_array = torch.stack(u_list).cpu().numpy()  # Convert to numpy for plotting
    #     plt.figure(figsize=(10, 6))
    #     # for i in range(u_array.shape[2]):  # Loop through all hidden units
    #     plt.plot([u_array[i, :, :] for i in range(u_array.shape[2])])
    #     plt.title('Membrane Potential (u) Over Time')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Membrane Potential (u)')
    #     plt.legend(loc='upper right')
    #     plt.grid(True)
    #     plt.savefig(f"{resultroot}/u_plot.png")
    #     plt.show()
        
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
        
        for t in range(x.size(1)):      
                hy, hz, u, spk = self.cell(x[:, t], hy, hz, u)
                hy_list.append(hy)
                hz_list.append(hz)
                u_list.append(u)
                spike_list.append(spk)
        return hy_list, hz_list, u_list, spike_list