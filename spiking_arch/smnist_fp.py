# $ python -m spiking_arch.smnist_fp --resultroot spiking_arch/results/ --sron
# $ python -m spiking_arch.smnist_fp --dataroot "MNIST/" --resultroot "spiking_arch/results/spiking_act" --sron --batch 16

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

from spiking_arch.lsm_baseline import LiquidRON
from spiking_arch.s_ron import SpikingRON
from spiking_arch.mixed_ron import MixedRON


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
parser.add_argument("--rho", type=float, default=9, help="ESN spectral radius")
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

parser.add_argument("--threshold", type=float, default=1.0, help="threshold")
parser.add_argument("--resistance", type=float, default=5.0, help="resistance")
parser.add_argument("--capacitance", type=float, default=3.0, help="capacitance")
parser.add_argument("--reset", type=float, default=-1.0, help="reset")
parser.add_argument("--rc", type=float, default=1.0, help="resistance x capacitance")
parser.add_argument("--bias", type=float, default=0.01, help="bias")


args = parser.parse_args()

if args.dataroot is None:
    warnings.warn("No dataroot provided. Using current location as default.")
    args.dataroot = os.getcwd()
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()
assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."

assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"



@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        output = model(images)[0][-1]
        if args.liquidron is None:
            activations.append(output[-1])
        else:
            activations.append(output)
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)



device = (
    torch.device("cuda")
    # if torch.cuda.is_available() and not args.cpu
    # else torch.device("cpu")
)
print('Using device: ', device)

n_inp = 1
n_out = 10

gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

train_accs, valid_accs, test_accs = [], [], []
for i in range(args.trials):
    if args.esn:
        model = DeepReservoir(
            n_inp,
            tot_units=args.n_hid,
            spectral_radius=args.rho,
            input_scaling=args.inp_scaling,
            connectivity_recurrent=int((1 - args.sparsity) * args.n_hid),
            connectivity_input=args.n_hid,
            leaky=args.leaky,
        ).to(device)
    elif args.ron:
        model = RandomizedOscillatorsNetwork(
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
        ).to(device)
    elif args.pron:
        model = PhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device
        ).to(device)
    elif args.mspron:
        model = MultistablePhysicallyImplementableRandomizedOscillatorsNetwork(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.inp_scaling,
            device=device
        ).to(device)
        
    elif args.sron:
        model = SpikingRON(
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
            args.rc,
            args.reset,
            args.bias,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
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
            args.rc,
            args.reset,
            args.bias,
            win_e=2.5,
            win_i=1.5,
            w_e=1,
            w_i=0.5,
            Ne=200,
            Ni=56,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device
        ).to(device)
    else:
        raise ValueError("Wrong model choice.")
    
    train_loader, valid_loader, test_loader = get_mnist_data(
        args.dataroot, args.batch, args.batch
    )
    
    activations, ys = [], []
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        ys.append(labels) 
        if args.liquidron:
            output, u, spk = model(images)
            activations.append(output)
            # print(output)
        else:
            output, velocity, u, spk = model(images)
            activations.append(output[-1])
            
    if not args.liquidron:# is None:
        output=torch.from_numpy(np.array(output, dtype=np.float32))
        u=torch.from_numpy(np.array(u, dtype=np.float32))   
        spk=torch.from_numpy(np.array(spk, dtype=np.float32)) 
        velocity=torch.from_numpy(np.array(velocity, dtype=np.float32))  
        plot_dynamics(output, velocity, u, spk, images, args.resultroot)
    activations = torch.cat(activations, dim=0).numpy()
    print('activations:', activations.shape)
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    # print('NaN in activations', np.isnan(activations).sum())  # Count NaNs

    # print('activations shape: ', activations.shape,'activations items shape: ', activations[-1].size(), '\nys shape: ', ys.shape, 'ys items shape: ', ys[-1].size())
    scaler = preprocessing.StandardScaler()#.fit(activations)
    activations = scaler.fit_transform(activations) #scaler.transform(activations) 
    classifier = LogisticRegression(max_iter=5000).fit(activations, ys)
    train_acc = test(train_loader, classifier, scaler)
    # valid_acc = test(valid_loader, classifier, scaler) #if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) #if args.use_test else 0.0
    # train_accs.append(train_acc)
    # valid_accs.append(valid_acc)
    test_accs.append(test_acc)
simple_plot(train_accs, valid_accs, test_accs, args.resultroot)


if args.ron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.pron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_PRON{args.resultsuffix}.txt"), "a")
elif args.mspron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_MSPRON{args.resultsuffix}.txt"), "a")
elif args.esn:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_ESN{args.resultsuffix}.txt"), "a")
elif args.sron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_SRON{args.resultsuffix}.txt"), "a")
elif args.liquidron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_LiquidRON{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")


ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    # f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
    # f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]}"
    # f"mean/std train: {np.mean(train_accs), np.std(train_accs)} "
    # f"mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} "
    f"mean/std test: {np.mean(test_accs), np.std(test_accs)}"
)
f.write(ar + "\n")
f.close()
