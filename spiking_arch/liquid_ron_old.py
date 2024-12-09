'''
The execution code you provided seems already compatible with the `LiquidStateMachineRON` model, with the key model type being switched in the following block:

```python
elif args.sron:
    model = SpikingRON(
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
```

However, to ensure compatibility and highlight a few considerations:

1. **Class Names and Parameters**:
    - If the `LiquidStateMachineRON` class is the intended model (you might want to rename `SpikingRON` to `LiquidStateMachineRON` in the import section of your execution code, assuming `SpikingRON` is a placeholder for this model).
    
    - It seems like there are some mismatched parameters, such as `win_e`, `win_i`, `w_e`, `w_i`, and others in the `LiquidStateMachineRON` model's initialization, which are not explicitly passed from the execution script. You will need to either pass these values from the command line interface (CLI) or remove them from the model class if not needed in the implementation.

2. **Model Adjustments**:
    - In `LiquidStateMachineRON`, certain parameters such as `Ne`, `Ni`, and others are set manually or are pulled from `kwargs`, but in the execution code, they seem to be missing as input arguments. You might want to ensure these are either set by default values or passed from the execution script.
    
    - It's also important to note that in the `train()` method, there's an expectation of `states.T @ states` being used to compute the regularized or non-regularized output. This might require ensuring that the data structure for the `states` variable (output of the `forward` pass) matches the expected shape in your execution code.

3. **Handling Data**:
    - The execution code processes batches from the MNIST dataset with a loop to gather activations and spikes. Ensure that the model output matches what is expected: the code references a `forward()` method that returns `output, velocity, u, spk` (spike data). If `LiquidStateMachineRON`'s `forward()` only returns `states`, you will need to adjust the execution code to handle just that output or modify `forward()` to return additional information (like `velocity`, `u`, etc.) for consistency with the execution script.

In summary:
- If you're working with `LiquidStateMachineRON`, ensure that it is correctly imported and the expected parameters match the command-line arguments. 
- The structure of the output in `forward()` should align with how the results are expected by the execution script (activation, spikes, and other data). If this aligns, then no further changes are required.

'''

from typing import (
    List,
    Literal,
    Tuple,
    Union,
)

import torch
from torch import nn
import numpy as np
from numpy import ones
import matplotlib.pyplot as plt

from acds.archetypes.utils import (
    get_hidden_topology,
    spectral_norm_scaling,
)


class LiquidRON(nn.Module):

    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        rho: float,
        input_scaling: float,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cpu",
        win_e=2,
        win_i = 1,
        w_e=0.5,
        w_i=0.2,
        reg=None
    ):
        """Initialize the RON model.

        Args:
            n_inp (int): Number of input units.
            n_hid (int): Number of hidden units.
            dt (float): Time step.
            gamma (float or tuple): Damping factor. If tuple, the damping factor is
                randomly sampled from a uniform distribution between the two values.
            epsilon (float or tuple): Stiffness factor. If tuple, the stiffness factor
                is randomly sampled from a uniform distribution between the two values.
            rho (float): Spectral radius of the hidden-to-hidden weight matrix.
            input_scaling (float): Scaling factor for the input-to-hidden weight matrix.
                Wrt original paper here we initialize input-hidden in (0, 1) instead of (-2, 2).
                Therefore, when taking input_scaling from original paper, we recommend to multiply it by 2.
            topology (str): Topology of the hidden-to-hidden weight matrix. Options are
                'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz'. Default is
                'full'.
            reservoir_scaler (float): Scaling factor for the hidden-to-hidden weight
                matrix.
            sparsity (float): Sparsity of the hidden-to-hidden weight matrix.
            device (str): Device to run the model on. Options are 'cpu' and 'cuda'.
        """
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        
        self.win_e=win_e
        self.win_i=win_i
        self.w_e=w_e
        self.w_i=w_i
        self.Ne=800     #kwargs.get('ne')
        self.Ni=200     #kwargs.get('ni')
        self.reg = reg
        self.readout = np.random.rand(self.Ne + self.Ni)
        self.re = np.random.rand(self.Ne)
        self.ri = np.random.rand(self.Ni)
        self.a = torch.cat((0.02*ones(self.Ne), 0.02+0.08*self.ri))
        self.b = torch.cat((0.2*np.ones(self.Ne), 0.25-0.05*self.ri))
        self.c = torch.cat((-65+15*self.re**2, -65*np.ones(self.Ni)))
        self.d = torch.cat((8-6*self.re**2, 2*np.ones(self.Ni)))
        self.v = -65*np.ones(self.Ne+self.Ni)  # Initial values of v
        self.u = self.v*self.b
        self.U = torch.cat((self.win_e*np.ones(self.Ne), self.win_i*np.ones(self.Ni)))
        self.S = torch.cat((self.w_e*np.random.rand(self.Ne+self.Ni, self.Ne), -self.w_i*np.random.rand(self.Ne+self.Ni, self.Ni)), axis=1)

        h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        h2h = spectral_norm_scaling(h2h, rho)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        x2h = torch.rand(n_inp, n_hid) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)
        
    def forward(self, data):
        u,v = torch.tensor(self.u, dtype=torch.float32), torch.tensor(self.v, dtype=torch.float32) #self.u, self.v
        firings = []  # spike timings
        states = []  # here we construct the matrix of reservoir states
        for t in range(len(data)):  # simulation of 1000 ms
            I = torch.tensor((data[t] * self.U), dtype=torch.float32)
            fired = torch.where(v >= 30)[0]  # indices of spikes
            firings.append(np.column_stack((t+np.zeros_like(fired), fired)))
            v[fired] = self.c[fired]
            u[fired] = u[fired] + self.d[fired]
            I = I + np.sum(self.S[:, fired], axis=1)
            print('Checkpoint for error \nv: ', v.shape, 'u: ', u.shape, 'I: ', I.shape)
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # step 0.5 ms
            # v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # for numerical stability
            u = u + self.a*(self.b*v - u)
            states.append(v >= 30)

        firings = torch.cat(firings)

        # in the end states is 1000 x number of time steps
        return states, v, u, firings 
    
    def train(self, data, target):
        states = self.forward(data)
        if self.reg is not None:
            self.readout = np.linalg.pinv(states.T @ states + np.eye(states.shape[1]) * reg) @ states.T @ target
        else:
            self.readout = np.linalg.pinv(states) @ target
            
        return states @ self.readout, v, u, firings 