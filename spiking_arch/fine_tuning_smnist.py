# $ python -m spiking_arch.fine_tuning_smnist --dataroot "MNIST/" --resultroot "spiking_arch/results/tuning_sron" --sron --batch **
###
# $ nohup python3 -m spiking_arch.fine_tuning_smnist --dataroot "MNIST/" --resultroot "spiking_arch/results/tuning_sron" --sron --batch 256  > out.log >&grid&
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

from spiking_arch.lsm_baseline import LiquidRON
from spiking_arch.s_ron import SpikingRON
from spiking_arch.mixed_ron import MixedRON

from acds.benchmarks import get_mnist_data

from spiking_arch.snn_utils import *

torch.cuda.empty_cache()

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
    ## ORIGINAL
    default=4.7,
    # default = 0.51,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument(
    "--gamma_range",
    type=float,
    default=2.7,
    # default=1.0,
    help="y controle parameter <gamma> of the coRNN",
)
parser.add_argument(
    "--epsilon_range",
    type=float,
    default=4.7,
    # default= 0.5,
    help="z controle parameter <epsilon> of the coRNN",
)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--esn", action="store_true")
parser.add_argument("--ron", action="store_true")
parser.add_argument("--pron", action="store_true")
parser.add_argument("--mspron", action="store_true")

parser.add_argument("--sron", action="store_true")
parser.add_argument("--liquidron", action="store_true")
parser.add_argument("--mixron", action="store_true")

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

parser.add_argument("--threshold", type=float, default=1.0, help="spiking ron models threshold")
parser.add_argument("--resistance", type=float, default=5.0, help="resistance (spiking n.)")
parser.add_argument("--capacitance", type=float, default=5.e-3, help="capacitance (spiking n.)")
parser.add_argument("--reset", type=float, default=-1.0, help="spiking ron models reset")
parser.add_argument("--bias", type=float, default=0.01, help="bias")
parser.add_argument("--perc", type=float, default=0.5, help="percentage of neurons")


args = parser.parse_args()
# Define the parameter grid
param_grid = {
    # "dt": [0.02, 0.05],  # Example values for time step
    # "gamma": [2.5, 2.9],  # Range of gamma values
    # "epsilon": [4.5, 4.9],  # Range of epsilon values
    # "rho": [0.9, 0.99],  # Spectral radius
    # "inp_scaling": [0.5, 0.8, 1.2],  # Input scaling
    'rc':[0.5, 2, 5, 7], # resistance x capacitance
    "threshold": [0.1, 0.09, 0.05],
    # "resistance": [3.0, 5.0, 7.0],
    # "capacitance": [3e-3, 5e-3, 7e-3],
    "reset": [0.001, 0.004],#-1, 0.001, 0.005], # initial membrane potential 
    "bias": [0, 0.001, 0.005, 0.01, 0.1, 0.25],
    
    # Input weights
    # "win_e": [1.0, 1.5, 2.0, 2.5, 3.0],  
    # "win_i": [0.5, 1.0, 1.5, 2.0],

    # # Recurrent weights
    # "w_e": [0.5, 1.0, 1.5, 2.0],  
    # "w_i": [0.2, 0.5, 0.8, 1.0]
    }

# Convert grid to list of combinations
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

best_params = None
best_valid_acc = 0.0
best_train_acc = 0.0
best_test_acc = 0.0


device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)
print("Using device ", device)
n_inp = 1
n_out = 10

gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

all_acc = []

@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[0][-1]
        activations.append(output)
        ys.append(labels)
    # activations = torch.cat(activations, dim=0).numpy()
    activations = torch.cat(activations, dim=0).cpu().detach().numpy() 
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

# @torch.no_grad()
for param_set in tqdm(param_combinations, desc="Grid Search"):
    # Unpack parameters
    params = dict(zip(param_names, param_set))
    print(f"Testing parameters: {params}")
    
    if args.sron:
        model = SpikingRON(
        n_inp,
        args.n_hid,
        args.dt,
        # params["dt"],
        # (params["gamma"] - args.gamma_range / 2.0, params["gamma"] + args.gamma_range / 2.0),
        (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0),
        (args.epsilon - args.epsilon_range / 2.0, args.epsilon + args.epsilon_range / 2.0),
        # (params["epsilon"] - args.epsilon_range / 2.0, params["epsilon"] + args.epsilon_range / 2.0),
        ##do not 
        args.rho,
        args.inp_scaling,
        # params["threshold"],
        args.threshold,
        # params["resistance"],
        # params["capacitance"], 
        # args.resistance,
        # args.capacitance, 
        params['rc'],       
        # params["reset"], 
        args.reset, 
        params['bias'],
        topology=args.topology,
        sparsity=args.sparsity,
        reservoir_scaler=args.reservoir_scaler,
        device=device
    ).to(device)
        
    elif args.liquidron:
        model = LiquidRON(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.rho,
            args.inp_scaling,
            # spiking
            args.threshold,
            # args.resistance,
            # args.capacitance,
            params['rc'],       
            #args.rc,
            args.reset,
            # args.bias,
            params['bias'],
            params['win_e'],
            # win_e=2.5,
            params['win_i'],
            # win_i=1.5,
            params['w_e'],
            params['w_i'], 
            # w_e=1,
            # w_i=0.5,
            Ne=200,
            Ni=56,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device
        ).to(device)
        
    elif args.mixron:
        model = MixedRON(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.rho,
            args.inp_scaling,
            #add last things here
            # args.threshold,
            params['threshold'],        
            # args.resistance,
            # args.capacitance,
            params['rc'],       
            # args.rc,
            params['reset'],       
            # args.reset,
            params['bias'],       
            # args.bias,
            args.perc,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
        ).to(device) 
    else:
        raise ValueError("Wrong model choice.")
    # Create the model with current parameters
    
    # Train and validate the model
    train_loader, valid_loader, test_loader = get_mnist_data(args.dataroot, args.batch, args.batch)
    activations, ys = [], []

    for images, labels in train_loader:
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[0][-1]
        activations.append(output)
        ys.append(labels)

    # activations = [torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in activations]
    # activations = torch.cat(activations, dim=0).numpy()
    activations = torch.cat(activations, dim=0).cpu().detach().numpy() 
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    classifier = LogisticRegression(max_iter=5000).fit(activations, ys)

    # train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler)
    # test_acc = test(test_loader, classifier, scaler)
    
    all_acc.append(valid_acc)
    # print(f"Train Acc: {train_acc:.2f}, Valid Acc: {valid_acc:.2f}, Test Acc: {test_acc:.2f}")
    print(f"Valid Acc: {valid_acc:.2f}")
    
    # Update best parameters if validation accuracy improves
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        # best_train_acc = train_acc
        # best_test_acc = test_acc
        best_params = params
        
acc_table(param_names, param_combinations, all_acc, args.resultroot, 'smnist')

if args.sron:
    f = open(os.path.join(args.resultroot, f"finetune_sMNIST_SRON{args.resultsuffix}.txt"), "a")
elif args.liquidron:
    f = open(os.path.join(args.resultroot, f"finetune_sMNIST_LiquidRON{args.resultsuffix}.txt"), "a")
elif args.mixron:
    f = open(os.path.join(args.resultroot, f"finetune_sMNIST_MixedRON{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")


ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"Best Parameters: {best_params}"
    f"Best Validation Accuracy: {best_valid_acc:.2f}"
    # f"Best Train Accuracy: {best_train_acc:.2f}"
    # f"Best Test Accuracy: {best_test_acc:.2f}"
    
)
f.write(ar + "\n")
f.close()

# Report best parameters and performance
print(f"Best Parameters: {best_params}")
print(f"Best Validation Accuracy: {best_valid_acc:.2f}")
