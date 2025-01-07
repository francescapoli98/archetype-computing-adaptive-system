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

from spiking_arch.liquid_ron import LiquidRON
from spiking_arch.s_ron_ import SpikingRON

from acds.benchmarks import get_mnist_data

from spiking_arch.snn_utils import *

# Argument parsing and setting up initial arguments (unchanged)
parser = argparse.ArgumentParser(description="training parameters")
# (Your parser code goes here...)

args = parser.parse_args()

# Assertions and initializations (unchanged)
if args.dataroot is None:
    warnings.warn("No dataroot provided. Using current location as default.")
    args.dataroot = os.getcwd()
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."

assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"

# Define the device
device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

n_inp = 1
n_out = 10

gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

train_accs, valid_accs, test_accs = [], [], []

# Iterate over the number of trials
for i in range(args.trials):
    # Model selection (same as before, based on args)
    if args.liquidron:
        model = LiquidRON(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.rho,
            args.inp_scaling,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
            ### Fine-tune parameters added
            win_e=2,
            win_i=1,
            w_e=0.5,
            w_i=0.2,
            reg=None,
        ).to(device)
    else:
        # Other model choices (same as before)
        pass

    # Data loading (same as before)
    train_loader, valid_loader, test_loader = get_mnist_data(
        args.dataroot, args.batch, args.batch
    )
    
    activations, ys, x = [], [], [] 
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output, velocity, u, spk = model(images)
        activations.append(output[-1])
        ys.append(labels) 
        x.append(images)
        break

    # Fine-tuning: freezing layers, updating specific ones
    # Freeze all layers except the output layer (for example):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.output_layer.parameters():  # Assuming the last layer is the output layer
        param.requires_grad = True
    
    # Set learning rates for different parts (e.g., lower for frozen layers):
    optimizer = torch.optim.Adam(
        [
            {"params": model.output_layer.parameters(), "lr": 1e-3},  # Fine-tuning output layer
            {"params": model.parameters(), "lr": 1e-5, "weight_decay": 1e-4},  # Freezing other layers
        ]
    )

    # Training loop
    model.train()
    for epoch in range(epochs):  # Define epochs variable
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            output, _, _, _ = model(images)
            loss = criterion(output, labels.to(device))  # Define criterion (e.g., cross-entropy)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations) 
    classifier = LogisticRegression(max_iter=5000).fit(activations, ys)
    train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler) #if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) #if args.use_test else 0.0
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)

# Plotting and saving results (unchanged)
simple_plot(train_accs, valid_accs, test_accs, args.resultroot)

# Logging results (unchanged)
if args.liquidron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_LiquidRON{args.resultsuffix}.txt"), "a")
