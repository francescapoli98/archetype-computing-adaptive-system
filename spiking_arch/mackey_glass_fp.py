import argparse
import warnings
import os
import numpy as np
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import Ridge

from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_mackey_glass

from spiking_arch.lsm_baseline import LiquidRON
from spiking_arch.s_ron import SpikingRON
from spiking_arch.mixed_ron import MixedRON


from spiking_arch.snn_utils import *

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="training parameters")

parser.add_argument("--dataroot", type=str,
                    help="Path to the folder containing the mackey_glass.csv dataset")
parser.add_argument("--resultroot", type=str)
parser.add_argument("--resultsuffix", type=str, default="", help="suffix to append to the result file name")
parser.add_argument(
    "--n_hid", type=int, default=256, help="hidden size of recurrent net"
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

parser.add_argument("--threshold", type=float, default=1, help="threshold")
# parser.add_argument("--resistance", type=float, default=5.0, help="resistance")
# parser.add_argument("--capacitance", type=float, default=3.0, help="capacitance")
parser.add_argument("--rc", type=float, default=5.0, help="tau")
parser.add_argument("--reset", type=float, default=0.01, help="reset")
parser.add_argument("--bias", type=float, default=0.0, help="bias")
parser.add_argument("--perc", type=float, default=0.5, help="percentage of neurons")


args = parser.parse_args()


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
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1).numpy()
    # activations = model(dataset)[0].cpu().numpy()
    if args.liquidron:
        output, spk = model(dataset)
    else:
        output, velocity, u, spk = model(dataset)
    # activations = output[:, washout:]
    activations = torch.stack(output, dim=1)[:, washout:]
    activations = activations.reshape(-1, args.n_hid).cpu()
    activations = scaler.transform(activations)
    # print('activations: \n', activations)
    predictions = classifier.predict(activations)
    # print('predictions: \n', predictions)
    error = criterion_eval(torch.from_numpy(predictions).float(), torch.from_numpy(target).float()).item()
    # print("Error:", error)
    # print("Predictions shape:", predictions.shape if hasattr(predictions, "shape") else type(predictions))
    return error, predictions


gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)


train_mse, valid_mse, test_mse = [], [], []
for i in range(args.trials):
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
            # spiking
            args.threshold,
            args.rc,
            args.reset,
            args.bias,
            win_e=1,
            win_i=0.5,
            w_e=1,
            w_i=0.5,
            Ne=200,
            Ni=56,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
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
                args.threshold,
                # args.resistance,
                # args.capacitance,
                args.rc,
                args.reset,
                args.bias,
                args.perc,
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

    dataset = train_dataset.reshape(1, -1, 1).to(device)
    # print(dataset.size())
    target = train_target.reshape(-1, 1).numpy()
    # activations = model(dataset)[0].cpu().numpy()
    if args.liquidron:
        output, spk = model(dataset)
    else:
        output, velocity, u, spk = model(dataset)
    # output, velocity, u, spk = model(dataset)
    print('output dim: ', output[0].size())
    activations = torch.stack(output, dim=1)
    print('activations size: ', activations.size())
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid).cpu()
    # scaler = preprocessing.StandardScaler().fit(activations)
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = Ridge(max_iter=1000).fit(activations, target)
    train_nmse, train_pred = test(train_dataset, train_target, classifier, scaler)
   
    mg_results(train_target, train_pred, train_nmse, args.resultroot, 'MG_train_pred.png')
    train_mse.append(train_nmse)
    print('test dataset len: ', test_dataset.size())
    test_nmse, test_pred = (
        test(test_dataset, test_target, classifier, scaler) #if args.use_test else 0.0
    )
    mg_results(test_target, test_pred, test_nmse, args.resultroot, 'MG_test_pred.png')
    test_mse.append(test_nmse)
    
     # valid_nmse = (
    #     test(valid_dataset, valid_target, classifier, scaler)        
    # if not args.use_test
    #     else 0.0
    # )
    # mg_results(val_target, val_pred, args.resultroot, 'MG_val_pred.png')
    # valid_mse.append(valid_nmse)

    
print('Train mse: ', [str(round(train_acc, 2)) for train_acc in train_mse],
      '\nTest mse: ', [str(round(test_acc, 2)) for test_acc in test_mse])

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
    f"train: {[str(round(train_acc, 2)) for train_acc in train_mse]} "
    # f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_mse]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_mse]}"
    f"mean/std train: {np.mean(train_mse), np.std(train_mse)} "
    # f"mean/std valid: {np.mean(valid_mse), np.std(valid_mse)} "
    f"mean/std test: {np.mean(test_mse), np.std(test_mse)}"
)
f.write(ar + "\n")
f.close()
