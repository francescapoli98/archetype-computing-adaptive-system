'''
Set of useful functions (mostly plots)
'''
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import numpy as np


def simple_plot(train_accs: np.ndarray, valid_accs: np.ndarray, test_accs: np.ndarray, resultroot: str):
    print('this is the simple plot')
    # Convert lists to numpy arrays for easy manipulation
    train_accs = np.array(train_accs)
    valid_accs = np.array(valid_accs)
    test_accs = np.array(test_accs)
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot the mean accuracy for each stage (train, valid, test)
    plt.plot(np.arange(len(train_accs)), train_accs, label='Train Accuracy', marker='o', linestyle='-', color='blue')
    plt.plot(np.arange(len(valid_accs)), valid_accs, label='Validation Accuracy', marker='s', linestyle='--', color='orange')
    plt.plot(np.arange(len(test_accs)), test_accs, label='Test Accuracy', marker='^', linestyle=':', color='green')
    
    # Add labels, title, and legend
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Trial for Each Experiment Stage')
    plt.legend()
    
    # Show grid for better readability
    plt.grid(True)
    
    # Save the figure as a PNG file
    plot_filename = os.path.join(resultroot, f"accuracy_plot.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    # Show the plot in a window
    # plt.show()
    
#######################################################################################################
#######################################################################################################
#######################################################################################################
def plot_dynamics(
    activations: torch.Tensor,
    velocity: torch.Tensor,
    membrane_potential: torch.Tensor,
    spikes: torch.Tensor,
    resultroot: str
):
    print('Plotting dynamics of hidden state, derivative, membrane potential, and spikes.')

    # Ensure the input tensors are in numpy format
    activations = activations.detach().cpu().numpy() if isinstance(activations, torch.Tensor) else activations
    velocity = velocity.detach().cpu().numpy() if isinstance(velocity, torch.Tensor) else velocity
    membrane_potential = membrane_potential.detach().cpu().numpy() if isinstance(membrane_potential, torch.Tensor) else membrane_potential
    spikes = spikes.detach().cpu().numpy() if isinstance(spikes, torch.Tensor) else spikes

    # Get the time steps (assuming they are aligned with the tensor shapes)
    time_steps = np.arange(activations.shape[0])  # Number of time steps (length of time axis)
    print('Time steps shape: ', time_steps.shape)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot the activations (hidden state hy) - Selecting the first unit (index 0)
    plt.subplot(2, 2, 1)
    plt.title('Hidden States (hy)')
    # Plot the first hidden unit over time
    plt.plot(time_steps, activations[:, 0, 0], label="Hidden State (hy)", color="blue", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the velocity (hidden state derivative hz) - Selecting the first channel (layer) and first unit
    plt.subplot(2, 2, 2)
    plt.title('Hidden States Derivatives (hz)')
    # Plot the first hidden unit of the first channel
    plt.plot(time_steps, velocity[:, 0, 0], label="Hidden State Derivative (hz)", color="orange", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the membrane potential (u) - Selecting the first channel (layer) and first unit
    plt.subplot(2, 2, 3)
    plt.title('Membrane Potential (u)')
    # Plot the first hidden unit of the first channel
    plt.plot(time_steps, membrane_potential[:, 0, 0], label="Membrane Potential (u)", color="green", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the spikes (as vertical lines at spike times) - Selecting the first channel (layer) and first unit
    plt.subplot(2, 2, 4)
    plt.title('Spikes')
    # Find the time steps where spikes occurred for the first unit of the first channel
    spike_times = time_steps[spikes[:, 0, 0] == 1]  # Identify the spike times
    print(f"Spike times: {spike_times}")
    # Plot spikes at those times
    plt.scatter(spike_times, membrane_potential[spike_times, 0, 0], color="red", label="Spikes", zorder=5, s=30)
    plt.xlabel('Time Step')
    plt.ylabel('Membrane Potential Value')

    # Finalize the plot
    plt.tight_layout()
    plt.savefig(f"{resultroot}/dynamics_plot.png")
    plt.close()
    plt.show()



# def plot_dynamics(hy: torch.Tensor, hz: torch.Tensor, u: torch.Tensor, spikes: torch.Tensor, resultroot: str):
#     print('this is the dynamics plot')
#     """
#     Plots the dynamics of the hidden state (hy), hidden state derivative (hz), 
#     membrane potential (u), and spikes all in one plot.

#     Args:
#         hy (torch.Tensor): Hidden state over time (n_time_steps, n_hid).
#         hz (torch.Tensor): Hidden state derivative over time (n_time_steps, n_hid).
#         x (torch.Tensor): Input over time (n_time_steps, n_inp).
#         u (torch.Tensor): Membrane potential over time (n_time_steps, n_hid).
#         spikes (torch.Tensor): Spike train over time (n_time_steps, n_hid).
#         resultroot (str): Path to save the plot.
#     """
#     # Convert tensors to numpy arrays
#     hy = hy.detach().cpu().numpy() if isinstance(hy, torch.Tensor) else hy
#     hz = hz.detach().cpu().numpy() if isinstance(hz, torch.Tensor) else hz
#     u = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else u
#     spikes = spikes.detach().cpu().numpy() if isinstance(spikes, torch.Tensor) else spikes
#     # x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
#     time_steps = np.arange(hy.shape[0])
#     plt.figure(figsize=(10, 6))

#     # Plot hy (hidden state) and hz (derivative of hidden state)
#     plt.plot(time_steps, hy[:, 0], label="Hidden State (hy)", color="blue", linestyle='-', linewidth=1)
#     plt.plot(time_steps, hz[:, 0], label="Hidden State Derivative (hz)", color="orange", linestyle='-', linewidth=1)

#     # Plot membrane potential u
#     plt.plot(time_steps, u[:, 0], label="Membrane Potential (u)", color="green", linestyle='-', linewidth=1)

#     # Plot spikes (as vertical lines at spike time points)
#     spike_times = time_steps[spikes[:, 0] == 1]  # Identify the spike times
#     plt.scatter(spike_times, u[spike_times, 0], color="red", label="Spikes", zorder=5, s=30)

#     # Adding labels and title
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')
#     plt.title('Dynamics of hy, hz, u and Spikes')

#     # Add legend
#     plt.legend()

#     # Save the plot
#     plt.tight_layout()
#     plt.savefig(f"{resultroot}/dynamics_plot.png")
#     plt.show()



# def plot_dynamics(hy: torch.Tensor, hz: torch.Tensor, x: torch.Tensor, u_mem: torch.Tensor, spikes: torch.Tensor, resultroot: str):
#     """Plot hy, hz, membrane potential (u) and spike trains for 2D data."""
    
#     # Ensure data is 2D (batch_size, time_steps)
#     if hy.ndimension() != 2:
#         raise ValueError("Data must be 2D (batch_size, time_steps).")
    
#     # Number of samples in the batch
#     batch_size = hy.size(0)
    
#     # Create a figure for plotting
#     plt.figure(figsize=(15, 10))
    
#     # Plot hidden state (hy) for the first batch (or average across batches)
#     plt.subplot(2, 2, 1)
#     plt.plot([y for y in hy[0].cpu().numpy()])  # Plot first sample in the batch
#     plt.title("Hidden State (hy) - Sample 0")
#     plt.xlabel("Time Steps")
#     plt.ylabel("hy (hidden state)")

#     # Plot hidden state derivative (hz)
#     plt.subplot(2, 2, 2)
#     plt.plot([z for z in hz[0].cpu().numpy()])  # Plot first sample in the batch
#     plt.title("Hidden State Derivative (hz) - Sample 0")
#     plt.xlabel("Time Steps")
#     plt.ylabel("hz (hidden state derivative)")
    
#     # Plot membrane potential (u) for the first batch (or average across batches)
#     plt.subplot(2, 2, 3)
#     plt.plot([u for u in u_mem[0].cpu().numpy()])  # Plot first sample in the batch
#     plt.title("Membrane Potential (u) - Sample 0")
#     plt.xlabel("Time Steps")
#     plt.ylabel("u (membrane potential)")

#     # Plot spike train
#     plt.subplot(2, 2, 4)
#     plt.plot([s for s in spikes[0].cpu().numpy()])  # Plot first sample in the batch
#     plt.title("Spike Train - Sample 0")
#     plt.xlabel("Time Steps")
#     plt.ylabel("Spikes")

#     # Tight layout for better spacing
#     plt.tight_layout()

#     # Save the plot to the result directory
#     plot_filename = os.path.join(resultroot, f"dynamics_plot.png")
#     plt.savefig(plot_filename)
#     print(f"Plot saved as {plot_filename}")
    
#     # Display the plot
#     plt.show()
