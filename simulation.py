import torch
import torch.nn as nn
import numpy as np
import time, sys, random
import matplotlib.pyplot as plt
import cv2
from torch.profiler import profile, record_function, ProfilerActivity

N = 128 #lattice size N x N
temperature = 2.27 #temperature
device = 'cuda' #choose device (CPU or CUDA)
batch = 1 #batch size

l = int(N/2)
b = 1

import network
model = network.Ising().to(device).eval()

lattice = torch.ones(size=(batch,1,N,N)).float().to(device)
cb1 = torch.Tensor(np.kron([[0, 1] * l, [1, 0] * l] * l, np.ones((1, 1)))).to(device)
cb2 = torch.Tensor(np.kron([[1, 0] * l, [0, 1] * l] * l, np.ones((1, 1)))).to(device)
ones = torch.ones(lattice.shape, device=device)

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    energy = model(lattice)

    #checkerboard - update color 1
    new_state = cb2*lattice.data + cb1*(2*torch.randint(2, size=(batch,1,N,N),device=device)-1).float()
    new_energy = model(new_state)
    deltaE = new_energy - energy
    p = torch.exp(-deltaE/temperature)
    p = torch.where(p > 1.0, ones, p)
    r = torch.rand(size=(1,1,N,N), device=device)
    p = (p-r)*cb1
    lattice.data = torch.where(p > 0, new_state, lattice.data)

    #checkerboard - update color 2
    energy = model(lattice)
    new_state = cb1*lattice.data + cb2*(2*torch.randint(2, size=(batch,1,N,N),device=device)-1).float()
    new_energy = model(new_state)
    deltaE = new_energy - energy
    p = torch.exp(-deltaE/temperature)
    p = torch.where(p > 1.0, ones, p)
    r = torch.rand(size=(1,1,N,N), device=device)
    p = (p-r)*cb2
    lattice.data = torch.where(p > 0, new_state, lattice.data)

    #show lattice as a image
    img = lattice.cpu().numpy()[0][0]
    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
    cv2.imshow('frame',img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        sys.exit()