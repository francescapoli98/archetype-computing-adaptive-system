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

# from spiking_arch.liquid_ron import LiquidRON
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

# parser.add_argument("--threshold", type=float, default=0.008, help="")
# parser.add_argument("--resistance", type=float, default=5.0, help="ESN input scaling")
# parser.add_argument("--capacity", type=float, default=5.3, help="ESN input scaling")
# parser.add_argument("--reset", type=float, default=1.0, help="ESN input scaling")


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
        activations.append(output)
        ys.append(labels)
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)



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
            #add last things here
            args.
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
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
            ### FINE TUNE THESE (test to see if it works)
            win_e=2,
            win_i=1,
            w_e=0.5,
            w_i=0.2,
            reg=None,
        ).to(device)

    else:
        raise ValueError("Wrong model choice.")
    
    train_loader, valid_loader, test_loader = get_mnist_data(
        args.dataroot, args.batch, args.batch
    )
    
    activations, ys, x = [], [], [] 
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        # print('original image shape: ', images.shape)
        images = images.view(images.shape[0], -1).unsqueeze(-1)
        # print('x (images) shape after unsqueeze: ', type(images), images.shape)
        output, velocity, u, spk = model(images)
        # print('mem pot 1: ', u[:,0,0],'\n mem pot 2: ', u[0,:,0], '\n mem pot 3: ', u[0,0,:] ) 
        # print('output shape: ', output.size(), 'output[-1]: ', output[-1].size(), '\nvelocity shape: ', velocity.size(), '\nu shape: ', u.size(), '\nspk shape: ', spk.size())
        # print('AFTER CALLING THE MODEL\noutput: ', output.size(), '\n velocity: ', velocity.size(), '\nu: ', u.size(), '\nspikes: ', spk.size())
        activations.append(output[-1])
        ys.append(labels) 
        x.append(images)
        break

    output=torch.from_numpy(np.array(output, dtype=np.float32))    
    velocity=torch.from_numpy(np.array(velocity, dtype=np.float32))    
    u=torch.from_numpy(np.array(u, dtype=np.float32))   
    # print('torch mem pot 1: ', u[:,0,0],'\ntorch mem pot 2: ', u[0,:,0], '\ntorch mem pot 3: ', u[0,0,:] ) 
    spk=torch.from_numpy(np.array(spk, dtype=np.float32)) 
    # print('activations shape: ', activations.shape, '\nvel_p shape: ', vel_p.shape, '\nmem_p shape: ', mem_p.shape, '\nspk_p shape: ', spk_p.shape)
    # print('datatypes and shapes: \n velocity: ', type(velocity), velocity,size(), '\n membrane potential: ', type(u), u.size(), '\n spikes: ', type(spk), spk.size())
    plot_dynamics(output, velocity, u, spk, images, args.resultroot) 
    activations = torch.cat(activations, dim=0).numpy()
    print('activations:', activations.shape, type(activations))
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    # print('activations shape: ', activations.shape,'activations items shape: ', activations[-1].size(), '\nys shape: ', ys.shape, 'ys items shape: ', ys[-1].size())
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations) 
    classifier = LogisticRegression(max_iter=5000).fit(activations, ys)
    train_acc = test(train_loader, classifier, scaler)
    valid_acc = test(valid_loader, classifier, scaler) #if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) #if args.use_test else 0.0
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    test_accs.append(test_acc)
# print(f'membrane potential: {u[:, 0, 0]}')
# print(f'membrane potential shape: {u.size()}')
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
elif args.lron:
    f = open(os.path.join(args.resultroot, f"sMNIST_log_LiquidRON{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")


ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
    f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]}"
    f"mean/std train: {np.mean(train_accs), np.std(train_accs)} "
    f"mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} "
    f"mean/std test: {np.mean(test_accs), np.std(test_accs)}"
)
f.write(ar + "\n")
f.close()
