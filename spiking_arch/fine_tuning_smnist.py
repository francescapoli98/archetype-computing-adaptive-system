from itertools import product
import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from spiking_arch.s_ron import SpikingRON

from acds.benchmarks import get_mnist_data

from spiking_arch.snn_utils import *


parser = argparse.ArgumentParser(description="training parameters")
parser.add_argument("--dataroot", type=str)
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=256, help="hidden size of recurrent net"
)
parser.add_argument("--batch", type=int, default=16, help="batch size")
parser.add_argument(
    "--dt", type=float, default=0.042, help="step size <dt> of the coRNN"
)
parser.add_argument(
    "--gamma", type=float, default=2.7, help="y controle parameter <gamma> of the coRNN"
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=4.7,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument(
    "--gamma_range",
    type=float,
    default=2.7,
    help="y controle parameter <gamma> of the coRNN",
)
parser.add_argument(
    "--epsilon_range",
    type=float,
    default=4.7,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--pron", action="store_true")
parser.add_argument("--mspron", action="store_true")

parser.add_argument("--sron", action="store_true")
parser.add_argument("--liquidron", action="store_true")

parser.add_argument("--inp_scaling", type=float, default=1.0, help="ESN input scaling")
parser.add_argument("--rho", type=float, default=0.99, help="ESN spectral radius")
parser.add_argument("--leaky", type=float, default=1.0, help="ESN spectral radius")
parser.add_argument("--use_test", action="store_true")
parser.add_argument(
    "--trials", type=int, default=1, help="How many times to run the experiment"
)
parser.add_argument(
    "--topology",
    type=str,
    default="full",
    choices=["full", "ring", "band", "lower", "toeplitz", "orthogonal"],
    help="Topology of the reservoir",
)
parser.add_argument(
    "--sparsity", type=float, default=0.0, help="Sparsity of the reservoir"
)
parser.add_argument(
    "--reservoir_scaler",
    type=float,
    default=1.0,
    help="Scaler in case of ring/band/toeplitz reservoir",
)

args = parser.parse_args()
# Define the parameter grid
param_grid = {
    "dt": [0.02, 0.04, 0.06],  # Example values for time step
    "gamma": [2.5, 2.7, 2.9],  # Range of gamma values
    "epsilon": [4.5, 4.7, 4.9],  # Range of epsilon values
    "rho": [0.8, 0.9, 0.99],  # Spectral radius
    "inp_scaling": [0.8, 1.0, 1.2]  # Input scaling
}

# Convert grid to list of combinations
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

best_params = None
best_valid_acc = 0.0

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


@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[0][-1]
        activations.append(output)
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)


for param_set in tqdm(param_combinations, desc="Grid Search"):
    # Unpack parameters
    params = dict(zip(param_names, param_set))
    print(f"Testing parameters: {params}")

    # Create the model with current parameters
    model = SpikingRON(
        n_inp,
        args.n_hid,
        params["dt"],
        (params["gamma"] - args.gamma_range / 2.0, params["gamma"] + args.gamma_range / 2.0),
        (params["epsilon"] - args.epsilon_range / 2.0, params["epsilon"] + args.epsilon_range / 2.0),
        params["rho"],
        params["inp_scaling"],
        topology=args.topology,
        sparsity=args.sparsity,
        reservoir_scaler=args.reservoir_scaler,
        device=device,
    ).to(device)

    # Train and validate the model
    train_loader, valid_loader, test_loader = get_mnist_data(args.dataroot, args.batch, args.batch)
    activations, ys = [], []

    for images, labels in train_loader:
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[0][-1]
        activations.append(output)
        ys.append(labels)

    activations = torch.cat(activations, dim=0).numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    classifier = LogisticRegression(max_iter=5000).fit(activations, ys)

    train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler)
    test_acc = test(test_loader, classifier, scaler)

    print(f"Train Acc: {train_acc:.2f}, Valid Acc: {valid_acc:.2f}, Test Acc: {test_acc:.2f}")

    # Update best parameters if validation accuracy improves
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_params = params

# Report best parameters and performance
print(f"Best Parameters: {best_params}")
print(f"Best Validation Accuracy: {best_valid_acc:.2f}")