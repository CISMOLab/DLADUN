import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from .mssa import MST
from .fourier2 import DAFNet

class EstimatorBranch(nn.Module):
    def __init__(self,dim) -> None:
        super(EstimatorBranch,self).__init__()
        self.GenWeight = nn.Sequential(
            nn.Conv2d(dim,dim,1,1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim,dim,1,1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim)
        )
    
    def forward(self,x):
        kernel = self.GenWeight(x)
        return kernel

class EstimatorBranchWithDegradation(nn.Module):
    def __init__(self,dim) -> None:
        super(EstimatorBranchWithDegradation,self).__init__()
        self.GenWeight = nn.Sequential(
            nn.Conv2d(dim*2,dim,1,1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim,dim,1,1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim)
        )
    
    def forward(self,x,degradation):
        assert(x.shape == degradation.shape)
        kernel = self.GenWeight(torch.cat([x,degradation],dim=1))
        return kernel

class InverseImaging(nn.Module):
    def __init__(self,dim) -> None:
        super(InverseImaging,self).__init__()
        self.Phi = EstimatorBranchWithDegradation(dim)
        self.InversePhi = EstimatorBranch(dim)
        self.InverseNoise = EstimatorBranchWithDegradation(dim)

    def forward(self,x,degradation):
        assert(x.shape == degradation.shape)
        phi = self.Phi(x,degradation)
        inversePhi = self.InversePhi(phi)
        inverseNoise = self.InverseNoise(x,degradation)
        inverse_x = inversePhi * ( x + inverseNoise )
        return inverse_x, phi

class GradientDescent(nn.Module):
    def __init__(self,dim) -> None:
        super(GradientDescent,self).__init__()
        self.Phi = EstimatorBranchWithDegradation(dim)
        self.InversePhi = EstimatorBranch(dim)
        self.Rho = EstimatorBranchWithDegradation(dim)

    def forward(self,x,degradation,x_bar):
        assert(x.shape == degradation.shape)
        phi = self.Phi(x,degradation)
        inversePhi = self.InversePhi(phi)
        rho = self.Rho(x,degradation)

        v = x - rho * inversePhi * (phi * x - x_bar)

        return v, phi

class Momentum(nn.Module):
    def __init__(self,dim) -> None:
        super(Momentum,self).__init__()
        self.sigma= EstimatorBranchWithDegradation(dim)
    
    def forward(self,xk,xk_1):
        assert(xk.shape == xk_1.shape)
        sigma = self.sigma(xk,xk_1)

        z = xk + sigma*(xk - xk_1)

        return z

class Phase(nn.Module):
    def __init__(self,dim) -> None:
        super(Phase,self).__init__()

        self.GP = GradientDescent(dim)
        self.Denoiser = DAFNet()
        # self.M = Momentum(dim)
    
    def forward(self,x,phi,x_bar):
        v , phi = self.GP(x,phi,x_bar)
        xk = self.Denoiser(v,phi)
        # z = self.M(xk,x)

        return xk, phi

class Net(nn.Module):
    def __init__(self,dim,stage) -> None:
        super(Net,self).__init__()

        self.Inint = InverseImaging(dim)
        self.Phases = nn.ModuleList([])
        for i in range(stage):
            self.Phases.append(Phase(dim))
    
    def forward(self,x_bar,mask):
        assert(x_bar.shape == mask.shape)
        x, phi = self.Inint(x_bar,mask) # z_0=x_0
        x_list = []
        x_list.append(x)
        for phase in self.Phases:
            x, phi =  phase(x, phi, x_bar)
            x_list.append(x)
        return x_list

if __name__=="__main__":
    x = torch.ones([2,28,256,256]).cuda()
    mask = torch.ones([2,28,256,256]).cuda()
    model = Net(28,2).cuda()
    y = model(x,mask)
    del y