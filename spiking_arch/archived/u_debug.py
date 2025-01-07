
import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork
)
from acds.archetypes.utils import (
    get_hidden_topology,
    spectral_norm_scaling,
)

from spiking_arch.liquid_ron import LiquidRON
from spiking_arch.s_ron_ import SpikingRON

from acds.benchmarks import get_mnist_data

from spiking_arch.snn_utils import *


def cell(input_data, hy, hz, u):
    
    dt = 0.042
    gamma = 2.7
    epsilon = 4.7
        
    #### to be changed to spiking dynamics
    h2h = get_hidden_topology(n_hid, 'full', 0.0, 1.0)
    h2h = spectral_norm_scaling(h2h, 0.99)
    h2h = nn.Parameter(h2h, requires_grad=False)
    
    x2h = torch.rand(n_inp, n_hid) * 1.0
    x2h = nn.Parameter(x2h, requires_grad=False)
    
    threshold = 0.008
    R = 5.0
    C = 5e-3 
    reset = 0.001 # initial membrane potential ## FINE TUNE THIS
    # print('u at the beginning of the function: ', old_u)
    condition = u > threshold 
    spike = torch.where(condition, torch.tensor(1.0), torch.tensor(0.0))
    print(spike == 0)
    old_u = u
    u[spike == 1] = 0.001  # Hard reset only for spikes
    # print('u after the hard reset: ', u)
    if torch.all(old_u == u):
        print('NOTHING IS CHANGING HERE')
    # tau = R * C
    # print('shapes for the u update: \n hy:', hy.shape, ' h2h: ', h2h.shape, '\n x: ', input_data.shape, ' x2h: ', x2h.shape)
    print('matrix multiplication results: ', torch.matmul(input_data, x2h))
    u_dot = (torch.matmul(hy, h2h) + torch.matmul(input_data, x2h)) # u dot (update)
    print(torch.where(u_dot > 0))
    u += (u_dot * (R*C))*dt # multiply to tau and dt
    # print('u updated: ', u)
    
    hz = hz + dt * ( 
        u 
        - gamma * hy 
        - epsilon * hz
    )

    hy = hy + dt * hz
    return hy, hz, u, spike


import matplotlib.pyplot as plt
import torch

def plot_u(u_list, t, resultroot, n_neurons_to_plot=10):
    """Plot the membrane potential u over time for a subset of neurons."""
    u_array = torch.stack(u_list).cpu().numpy()  # Convert to numpy for plotting
    # print('u array shape for plotting: ', u_array.shape)
    
    # Ensure we are plotting up to the first `n_neurons_to_plot` neurons (for clarity)
    # neurons_to_plot = min(n_neurons_to_plot, u_array.shape[2])  # Limit neurons to plot if more than requested
    
    # Plot membrane potential for each neuron over time
    plt.figure(figsize=(10, 6))
    for i in range(u_array.shape[2]):  # Loop through first n neurons
        plt.plot(u_array[:, :, i], label=f"Neuron {i+1}")  # Plot each neuron's potential
    
    plt.title('Membrane Potential (u) Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Membrane Potential (u)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f"{resultroot}/debug u/u_plot_{t}.png")
    # plt.show()
    plt.close()

    
resultroot = 'spiking_arch/results/'
dataroot='./'
train_loader, valid_loader, test_loader = get_mnist_data(dataroot, 16, 16)
device = (torch.device("cpu"))

n_inp = 1
n_hid = 256

for images, labels in tqdm(train_loader):
    images = images.to(device)
    # print('original image shape: ', images.shape)
    x = images.view(images.shape[0], -1).unsqueeze(-1)
    print('x size: ',x.size())
    # print('x (images) shape after unsqueeze: ', type(images), images.shape)
    hy_list, hz_list, u_list, spike_list = [], [], [], []
    hy = torch.zeros(x.size(0), n_hid).to(device) #x.size(0)
    hz = torch.zeros(x.size(0), n_hid).to(device)
    u = torch.zeros(x.size(0), n_hid).to(device)
    for t in range(x.size(1)):
        print('step ', t)
        hy, hz, u, spk = cell(x[:, t], hy, hz, u)
        # print('u shape: ', u.shape)
        hy_list.append(hy)#.clone().detach())
        hz_list.append(hz)#.clone().detach())
        u_list.append(u)#.clone().detach())
        spike_list.append(spk)
        # plot_u(u_list, t, resultroot)

