import numpy as np
from numpy import ones
import matplotlib.pyplot as plt

class LSM:
    """The LSM class implements a simulation of a spiking neural network using Leaky Integrate-and-Fire neurons. """
    def __init__(self, reg=None, **kwargs):
        # self.epochs=kwargs.get('epochs')#,"10")
        # self.units=kwargs.get('units')#,"100")
        self.win_e=kwargs.get('win_e')
        self.win_i=kwargs.get('win_i')
        self.w_e=kwargs.get('w_e')
        self.w_i=kwargs.get('w_i')
        self.Ne=800     #kwargs.get('ne')
        self.Ni=200     #kwargs.get('ni')
        self.reg = reg
        self.readout = np.random.rand(self.Ne + self.Ni)
        self.re = np.random.rand(self.Ne)
        self.ri = np.random.rand(self.Ni)
        self.a = np.concatenate((0.02*ones(self.Ne), 0.02+0.08*self.ri))
        self.b = np.concatenate((0.2*np.ones(self.Ne), 0.25-0.05*self.ri))
        self.c = np.concatenate((-65+15*self.re**2, -65*np.ones(self.Ni)))
        self.d = np.concatenate((8-6*self.re**2, 2*np.ones(self.Ni)))
        self.v = -65*np.ones(self.Ne+self.Ni)  # Initial values of v
        self.u = self.v*self.b
        self.U = np.concatenate((self.win_e*np.ones(self.Ne), self.win_i*np.ones(self.Ni)))
        self.S = np.concatenate((self.w_e*np.random.rand(self.Ne+self.Ni, self.Ne), -self.w_i*np.random.rand(self.Ne+self.Ni, self.Ni)), axis=1)
        
    def _simulation(self, data):
        u,v = self.u, self.v
        firings = []  # spike timings
        states = []  # here we construct the matrix of reservoir states
        for t in range(len(data)):  # simulation of 1000 ms
            I = data[t] * self.U
            fired = np.where(v >= 30)[0]  # indices of spikes
            firings.append(np.column_stack((t+np.zeros_like(fired), fired)))
            v[fired] = self.c[fired]
            u[fired] = u[fired] + self.d[fired]
            I = I + np.sum(self.S[:, fired], axis=1)
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # step 0.5 ms
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # for numerical stability
            u = u + self.a*(self.b*v - u)
            states.append(v >= 30)

        firings = np.concatenate(firings)

        # in the end states is 1000 x number of time steps
        return states
    
    def train(self, data, target):
        states = self._simulation(data)
        if self.reg is not None:
            self.readout = np.linalg.pinv(states.T @ states + np.eye(states.shape[1]) * reg) @ states.T @ target
        else:
            self.readout = np.linalg.pinv(states) @ target
            
        return states @ self.readout
    
        