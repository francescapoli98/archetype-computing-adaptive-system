# $ python -m spiking_arch.finetune_mg --dataroot "acds/benchmarks/raw/" --resultroot "spiking_arch/results/tuning_sron" --sron --batch 128 
###
# $ nohup python3 -m spiking_arch.finetune_mg --dataroot "acds/benchmarks/raw/" --resultroot "spiking_arch/results/tuning_sron" --sron --batch 128   > out.log >&sron_mg_grid& 
from itertools import product
import argparse
import warnings
import os
import numpy as np
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from tqdm import tqdm


from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_mackey_glass

from spiking_arch.lsm_baseline import LiquidRON
from spiking_arch.s_ron import SpikingRON

from spiking_arch.snn_utils import *

parser = argparse.ArgumentParser(description="training parameters")

parser.add_argument("--dataroot", type=str,
                    help="Path to the folder containing the mackey_glass.csv dataset")
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=100, help="hidden size of recurrent net"
)
parser.add_argument("--batch", type=int, default=30, help="batch size")
parser.add_argument("--lag", type=int, default=1, help="prediction lag")
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

parser.add_argument("--threshold", type=float, default=0.1, help="threshold")
# parser.add_argument("--resistance", type=float, default=5.0, help="resistance")
# parser.add_argument("--capacitance", type=float, default=3.0, help="capacitance")
parser.add_argument("--rc", type=float, default=5.0, help="tau")
parser.add_argument("--reset", type=float, default=0.01, help="reset")


args = parser.parse_args()


param_grid = {
    # "dt": [0.02, 0.05],  # Example values for time step
    # "gamma": [2.5, 2.9],  # Range of gamma values
    # "epsilon": [4.5, 4.9],  # Range of epsilon values
    # "rho": [0.9, 0.99],  # Spectral radius
    # "inp_scaling": [0.5, 0.8, 1.2],  # Input scaling
    'rc':[0.5, 2, 5, 7],
    "threshold": [0.009, 0.05, 0.5, 1],
    # "resistance": [3.0, 5.0, 7.0],
    # "capacitance": [3e-3, 5e-3, 7e-3],
    "reset": [-1, 0.001, 0.004], # initial membrane potential 
    "bias": [0.001, 0.005, 0.01, 0.05, 0.1, 0.25],
    
}


# Convert grid to list of combinations
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

best_params = None
best_valid_mse = 0.5
best_train_mse = 0.5
best_test_mse = 0.5


assert args.dataroot is not None, "No dataroot provided"
if args.resultroot is None:
    warnings.warn("No resultroot provided. Using current location as default.")
    args.resultroot = os.getcwd()

assert os.path.exists(args.resultroot), \
    f"{args.resultroot} folder does not exist, please create it and run the script again."

assert 1.0 > args.sparsity >= 0.0, "Sparsity in [0, 1)"

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)
print("Using device ", device)

n_inp = 1
n_out = 1
washout = 200


criterion_eval = torch.nn.L1Loss()

@torch.no_grad()
def test(dataset, target, classifier, scaler):
    # dataset = dataset.reshape(1, -1).to(device)
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1).numpy()
    # target = target.reshape(-1, 1).numpy()
    # activations = model(dataset)[0].cpu().numpy()
    output, velocity, u, spk = model(dataset)
    activations = torch.stack(output, dim=1)[:, washout:]
    activations = activations.reshape(-1, args.n_hid).cpu()
    activations = scaler.transform(activations)
    predictions = classifier.predict(activations)
    print('DIM prediction: ', predictions.shape, '\ntarget: ', target.shape)
    error = criterion_eval(torch.from_numpy(predictions).float(), torch.from_numpy(target).float()).item()
    return error, predictions


gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)


# train_mse, valid_mse, test_mse = [], [], []
all_mse = []


for param_set in tqdm(param_combinations, desc="Grid Search"):
    # Unpack parameters
    params = dict(zip(param_names, param_set))
    print(f"Testing parameters: {params}")
    if args.ron:
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
    elif args.sron:
        model = SpikingRON(
            n_inp,
            args.n_hid,
            args.dt,
            # params['dt'],
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
            params['reset'],
            # args.reset,
            params['bias'],
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
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
                topology=args.topology,
                sparsity=args.sparsity,
                reservoir_scaler=args.reservoir_scaler,
                device=device,
            ).to(device) 
    else:
        raise ValueError("Wrong model name")

    (
        (train_dataset, train_target),
        (valid_dataset, valid_target),
        (test_dataset, test_target),
    ) = get_mackey_glass(args.dataroot, lag=args.lag)

    # dataset = train_dataset.reshape(1, -1).to(device)
    dataset = train_dataset.reshape(1, -1, 1).to(device)
    target = train_target.reshape(-1, 1).numpy()
    # activations = model(dataset)[0].cpu().numpy()
    output, velocity, u, spk = model(dataset)
    # plot_dynam_mg(len(output),  
    #               torch.from_numpy(np.array(u, dtype=np.float32)),
    #               torch.from_numpy(np.array(spk, dtype=np.float32)),    
    #               args.resultroot)
    activations = torch.stack(output, dim=1)#.cpu().detach().numpy()
    activations = activations[:, washout:]#.cpu().detach().numpy()
    activations = activations.reshape(-1, args.n_hid).cpu()#.detach().numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    # activations = scaler.transform(activations)
    activations = scaler.transform(activations)
    classifier = Ridge(max_iter=1000).fit(activations, target)
    train_nmse, train_pred = test(train_dataset, train_target, classifier, scaler)
    
    
   
    # mg_results(train_target, train_pred, train_nmse, args.resultroot, 'MG_train_pred.png')
    # train_mse.append(train_nmse)
    # print('test dataset len: ', test_dataset.size())
    test_nmse, test_pred = test(test_dataset, test_target, classifier, scaler) 

    # mg_results(test_target, test_pred, test_nmse, args.resultroot, 'MG_test_pred.png')
    # test_mse.append(test_nmse)
    
    valid_nmse, valid_pred = test(valid_dataset, valid_target, classifier, scaler)        
    # if not args.use_test
    #     else 0.0
    
     
     
    all_mse.append(valid_nmse)
    print(f"Train MSE: {train_nmse:.2f}, Valid MSE: {valid_nmse:.2f}, Test MSE: {test_nmse:.2f}")

    # Update best parameters if validation accuracy improves
    if valid_nmse < best_valid_mse:
        best_valid_mse = valid_nmse
        best_train_mse = train_nmse
        best_test_mse = test_nmse
        best_params = params
        
acc_table(param_names, param_combinations, all_mse, args.resultroot, 'mg')


    
# print('Train mse: ', [str(round(train_, 2)) for train_acc in train_mse],
#       '\nTest mse: ', [str(round(test_acc, 2)) for test_acc in test_mse])

if args.ron:
    f = open(os.path.join(args.resultroot, f"MG_log_RON_{args.topology}{args.resultsuffix}.txt"), "a")
elif args.liquidron:
    f = open(os.path.join(args.resultroot, f"MG_log_LiquidRON{args.resultsuffix}.txt"), "a")
elif args.sron:
    f = open(os.path.join(args.resultroot, f"MG_log_SRON{args.resultsuffix}.txt"), "a")
elif args.mixron:
    f = open(os.path.join(args.resultroot, f"MG_log_MixedRON{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")


ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"Best Parameters: {best_params}"
    f"Best Validation Accuracy: {best_valid_mse:.2f}"
    f"Best Train Accuracy: {best_train_mse:.2f}"
    f"Best Test Accuracy: {best_test_mse:.2f}"
    
)
f.write(ar + "\n")
f.close()

# Report best parameters and performance
print(f"Best Parameters: {best_params}")
print(f"Best Validation MSE: {best_valid_mse:.2f}")
