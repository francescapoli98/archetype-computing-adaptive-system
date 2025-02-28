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



class LiquidRON(nn.Module):

    def __init__(
        self,
        n_inp: int,
        n_hid: 256,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        rho: float,
        input_scaling: 1.0,
        #spiking dynam
        threshold: float,
        # resistance: float,
        # capacitance: float,
        rc: float,
        reset: float,
        bias: float,
        #lsm
        win_e:int,
        win_i:int,
        w_e:float,
        w_i:float,
        Ne:int,    
        Ni:int,  
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cuda"
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
            
        #### to be changed to spiking dynamics
        # h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        h2h = np.concatenate((w_e*np.random.rand(Ne+Ni, Ne), 
                              - w_i*np.random.rand(Ne+Ni, Ni)), axis=1)   
        h2h = torch.tensor(h2h, dtype=torch.float32, device=self.device)  
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=True)  
        
        
        self.input_scaling = np.concatenate((win_e*np.ones(Ne), win_i*np.ones(Ni)))
        # print('INPUT SCALING DIM: ', self.input_scaling.shape)
        # x2h = torch.rand(n_inp, n_hid, device=self.device) * self.input_scaling
        # print('TENSORS FOR X2H: ', torch.rand(n_inp, n_hid).size(), torch.tensor(self.input_scaling).size())
        x2h = torch.rand(n_inp, n_hid, device=self.device) * torch.tensor(self.input_scaling, device=self.device)

        x2h = torch.tensor(x2h, dtype=torch.float32, device=self.device)  
        self.x2h = nn.Parameter(x2h, requires_grad=True)
        
        self.threshold = threshold 
        self.reset = reset # initial membrane potential ## FINE TUNE THIS
        self.rc = rc
        # self.reg = None  # Initialize regularization parameter.
        self.bias = bias        
        
        self.leaky = snn.Leaky(beta=0.9)
        
        
        
        
    def LIFcell(
        self, x: torch.Tensor, 
        hy: torch.Tensor, #hz: torch.Tensor, 
        u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and its derivative.

        Args:
            x (torch.Tensor): Input tensor ---> u(t) == I(t) external current
            hy (torch.Tensor): Current hidden state ----> position (y)
            hz (torch.Tensor): Current hidden state derivative ----> velocity (y'=z)
            u (torch.Tensor): Membrane potential 
        """
        # print(f"Type of u: {type(u)}, Type of threshold: {type(self.threshold)}")  # Check types
        
        spike = (u > self.threshold) * 1.0
        u[spike == 1] = self.reset  # Hard reset only for spikes
        # tau = R * C
        u_dot = - u + (torch.matmul(hy, self.h2h) + torch.matmul(x, self.x2h) + self.bias) # u dot (update) 
        # u += (u_dot * (self.R*self.C))*self.dt # multiply to tau and dt
        u = u + (u_dot * self.rc
                 #(self.R*self.C)
                 )*self.dt # multiply to tau and dt   
        # u += (self.dt / (self.R * self.C)) * u_dot
        # u[spike == 1] = self.reset  # hard reset only for spikes

        return u, spike

    
    # def readout_layer(self, states): #target
    #     # if self.reg is not None:
    #     #     self.readout = torch.linalg.pinv(states.T @ states + np.eye(states.shape[1]) * self.reg) @ states.T @ target
    #     # else:
    #     #     self.readout = torch.linalg.pinv(states) @ target
    #     #     # self.readout = torch.linalg.pinv(states.T @ states + torch.eye(states.shape[1]) * 1e-5) @ (states.T @ target)
        
    #     # return states @ self.readout
    #     return self.readout(states)
    
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

    def forward(self, x: torch.Tensor): 
        
        u_list, spike_list, hy_list = [], [], []
        # print('LSM PARAMS \nwin_e:', win_e, 'win_i:', win_i, 'w_e:', w_e, 'w_i:', w_i, 'Ne:', Ne, 'Ni:', Ni)
        u = torch.zeros(x.size(0), self.n_hid).to(self.device)
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device) #x.size(0)
        # print('input dim: ', x.size())
        for t in range(x.size(1)):
            u, spk = self.LIFcell(x[:, t], u, hy)
            # torch.any(spk_in == 1)
            hy = self.leaky(u)[0]
            # print('# spikes: ', (spk == 1).sum().item())
            hy_list.append(hy)
            u_list.append(u)
            spike_list.append(spk)
        # u_list, spike_list = torch.stack(u_list, dim=1).to(self.device), torch.stack(spike_list, dim=1).to(self.device)
        # self.readout = nn.Linear(self.n_hid, self.n_hid, bias=False).to(self.device)
        # readout = self.readout(u_list[:, -1])  # Shape: (batch_size, n_hid)
        return u_list, spike_list 