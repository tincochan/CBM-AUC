import torch 
import torch.nn as nn
from torch.nn import functional as F

class Ising(nn.Module):

    def __init__(self):
        super(Ising, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            )

        kernel = torch.zeros((3,3))
        kernel[0][1] = 1.0
        kernel[1][0] = 1.0
        kernel[1][2] = 1.0
        kernel[2][1] = 1.0

        self.features[0].weight.data[0][0] = kernel

    def forward(self, x):
        y = F.pad(x, (1, 1, 1, 1), mode='circular')
        x = -x*self.features(y)

        return x