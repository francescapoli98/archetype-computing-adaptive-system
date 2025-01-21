'''
Set of useful functions (mostly plots)
'''
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd



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
    x: torch.Tensor,
    resultroot: str
):
    print('Plotting dynamics of hidden state, derivative, membrane potential, and spikes.')

    # Ensure the input tensors are in numpy format
    activations = activations.detach().cpu().numpy() if isinstance(activations, torch.Tensor) else activations
    velocity = velocity.detach().cpu().numpy() if isinstance(velocity, torch.Tensor) else velocity
    membrane_potential = membrane_potential.detach().cpu().numpy() if isinstance(membrane_potential, torch.Tensor) else membrane_potential
    spikes = spikes.detach().cpu().numpy() if isinstance(spikes, torch.Tensor) else spikes
    # print('activations: ', activations.size(), ' velocity: ', velocity.size(), ' u: ', membrane_potential.size(), ' spikes: ', spikes.size())
    # Get the time steps (assuming they are aligned with the tensor shapes)
    time_steps = np.arange(len(activations))#.shape[1])  # Number of time steps (length of time axis)
    print('Time steps shape: ', time_steps.shape)

    # Create a plot
    plt.figure(figsize=(8, 10))
    
    # Plot the images (x) 
    plt.subplot(5, 1, 1)
    plt.title('Input (x)')
    # Plot the first hidden unit over time
    plt.plot(time_steps, x[0, :, 0], label="Input data (x)", color="purple", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the activations (hidden state hy) - Selecting the first unit (index 0)
    plt.subplot(5, 1, 2)
    plt.title('Hidden States (hy)')
    # Plot the first hidden unit over time
    plt.plot(time_steps, activations[:, 0, 0], label="Hidden State (hy)", color="blue", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the velocity (hidden state derivative hz) - Selecting the first channel (layer) and first unit
    plt.subplot(5, 1, 3)
    plt.title('Hidden States Derivatives (hz)')
    # Plot the first hidden unit of the first channel
    plt.plot(time_steps, velocity[:, 0, 0], label="Hidden State Derivative (hz)", color="orange", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the membrane potential (u) - Selecting the first channel (layer) and first unit
    plt.subplot(5, 1, 4)
    plt.title('Membrane Potential (u)')
    # Plot the first hidden unit of the first channel
    plt.plot(time_steps, membrane_potential[:, 0, 0], label="Membrane Potential (u)", color="green", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the spikes (as vertical lines at spike times) - Selecting the first channel (layer) and first unit
    plt.subplot(5, 1, 5)
    plt.title('Spikes')
    # Find the time steps where spikes occurred for the first unit of the first channel
    ##### SUBSTITUTE with basic scatter plot (time steps and 1-0 tensor of spikes)
    # spike_times = time_steps[spikes[:, 0, 0] == 1]  # Identify the spike times
    # print(f"Spike times: {spike_times}")
    # # Plot spikes at those times
    # plt.scatter(spike_times, membrane_potential[spike_times, 0, 0], color="red", label="Spikes", zorder=5, s=30)
    # spike_times = time_steps[spikes[:, 0, 0] == 1]  # Identify the spike times
    # print(f"Spike times: {spike_times}")
    # Plot spikes at those times
    plt.scatter(time_steps, spikes[:, 0, 0] == 1, color="red", label="Spikes", zorder=5, s=30)
    plt.xlabel('Time Step')
    plt.ylabel('Spike')

    # Finalize the plot
    plt.tight_layout()
    plt.savefig(f"{resultroot}/dynamics_plot.png")
    plt.close()
    plt.show()



def acc_table(param_combinations, accuracies, resultroot):
    """
    Create a table to visualize parameter combinations and their corresponding accuracies.

    Args:
        param_combinations (list of tuples): List of parameter combinations.
        accuracies (list of float): List of accuracies corresponding to the parameter combinations.
    """
    # Create a DataFrame from parameter combinations and accuracies
    param_names = ["dt", "rho", "inp_scaling", "threshold", "resistance", "capacity", "reset"]
    data = [list(comb) + [acc] for comb, acc in zip(param_combinations, accuracies)]
    df = pd.DataFrame(data, columns=param_names + ["Accuracy"])

    # Sort the DataFrame by accuracy (optional)
    df = df.sort_values(by="Accuracy", ascending=False)

    # Create a table plot
    fig, ax = plt.subplots(figsize=(12, min(1 + len(df) * 0.3, 20)))  # Adjust height for the number of rows
    ax.axis("off")
    ax.axis("tight")

    # Use the DataFrame as the table content
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#add8e6"] * (len(param_names) + 1),  # Light background for column headers
    )
    table.auto_set_font_size(True)
    # table.set_fontsize(10)
    table.auto_set_column_width(range(len(df.columns)))
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight="bold")  # Make header text bold
        cell.set_edgecolor("gray")  # Add a border color
        # cell.set_height(0.05)  # Adjust height
        # cell.set_width(0.15)  # Adjust width


    # Adjust layout
    fig.tight_layout()
    plt.savefig(f"{resultroot}/accuracies_gridsearch.png")
    plt.show()

def plot_hy(activations: torch.Tensor, x: torch.Tensor, resultroot: str):
    print("Plotting dynamics of the hidden state in comparison with the input.")

    # Convert tensors to numpy arrays if needed
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Validate dimensions
    assert activations.ndim == 3, f"Expected activations to be 3D, got {activations.ndim}D"
    assert x.ndim == 3, f"Expected x to be 3D, got {x.ndim}D"

    # Get time steps
    time_steps = np.arange(activations.shape[1])  # Time dimension is the second axis

    # Create plots
    plt.figure(figsize=(8, 10))

    # Plot input (x)
    plt.subplot(2, 1, 1)
    plt.title("Input (x)")
    plt.plot(time_steps, x[0, :, 0], label="Input Data", color="purple", linestyle="-", linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    # Plot hidden states (hy)
    plt.subplot(2, 1, 2)
    plt.title("Hidden States (hy)")
    plt.plot(time_steps, activations[0, :, 0], label="Hidden State", color="blue", linestyle="-", linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    # Finalize
    plt.tight_layout()
    os.makedirs(resultroot, exist_ok=True)
    plt.savefig(f"{resultroot}/hy_plot.png")
    plt.close()
    print(f"Plot saved to {resultroot}/hy_plot.png")

    
def plot_accuracy_fluctuations(param_combinations, accuracies):
    """
    Plots accuracy fluctuations for each set of values in the grid search.

    Args:
        param_combinations (list of tuples): The parameter combinations tested in the grid search.
        accuracies (list of floats): The validation accuracies corresponding to each parameter combination.
    """
    # Ensure the inputs are of the same length
    assert len(param_combinations) == len(accuracies), "Parameter combinations and accuracies must have the same length"

    # Create a unique identifier for each parameter combination
    param_labels = [" | ".join(f"{k}={v}" for k, v in zip(param_names, params)) for params in param_combinations]

    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracies)), accuracies, marker='o', linestyle='-', label='Validation Accuracy')

    # Label the points with their parameter combinations (optional for small grids)
    for i, label in enumerate(param_labels):
        plt.annotate(label, (i, accuracies[i]), fontsize=8, rotation=45, alpha=0.7, ha='right')

    plt.title("Accuracy Fluctuations Across Grid Search")
    plt.xlabel("Parameter Set Index")
    plt.ylabel("Validation Accuracy")
    plt.xticks(range(len(accuracies)), labels=range(len(accuracies)), rotation=45)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
