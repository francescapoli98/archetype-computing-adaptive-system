import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate

import numpy as np

u = torch.tensor([0.2,0,0.8,0,0.9])
#### ADD THE PRINTS HERE TO A OUTPUT FILE
spike = (u > 0.5) * 1.0
# hy was previously weighted with self.w and x was weighted with R --> now I use reservoir weights
if torch.any(u > 0.5):
    print('u has been reset')
old_u = u 
u[spike == 1] = 0  # Hard reset only for spikes
print(old_u, u)
# if torch.all(old_u == u):
#     print('NOT RESET!!')
# print("new u value:", u)
# # tau = R * C
# u_dot = - u + (torch.matmul(hy, self.h2h) + torch.matmul(x, self.x2h)) # u dot (update) 
# u += (u_dot * (self.R*self.C))*self.dt # multiply to tau and dt

print('spike:', spike, u) 