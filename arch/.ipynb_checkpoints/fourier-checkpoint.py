import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LocalFourierFilter(nn.Module):
    def __init__(self,dim,filer_h,filter_w, dynamic=False) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(dim,dim,1,1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim)
        )
        self.LocalConv2D = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim,dim,3,1,1)
        )
        self.dynamic = dynamic
        if self.dynamic:
            self.scale = nn.Parameter(torch.randn([1]) * 0.02)
            trunc_normal_(self.scale, std=.02)
        elif not self.dynamic:
            self.complex_weight = nn.Parameter(torch.randn(dim, filer_h, filter_w, 2, dtype=torch.float32) * 0.02)
            trunc_normal_(self.complex_weight, std=.02)

    def forward(self,x,filter_weight=None):

        ####### local
        x = self.embedding(x)

        ####### local brach
        x = self.LocalConv2D(x)
        ####### global
        x = x.to(torch.float32)
        B, C, a, b = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

        if self.dynamic:
            filter_weight = filter_weight.to(torch.float32) * self.scale
            if not filter_weight.shape[2:4] == x.shape[2:4]:
                size = x.shape[2:4]
                filter_weight = F.interpolate(filter_weight, size, mode='bilinear', align_corners=True)
                filter_weight = filter_weight.reshape(B,C,x.shape[2],x.shape[3],2)
            filter_weight = torch.view_as_complex(filter_weight.contiguous())
            x = x * filter_weight

        elif not self.dynamic:
            weight = self.complex_weight
            if not weight.shape[1:3] == x.shape[2:4]:
                weight = F.interpolate(weight.permute(3,0,1,2), size=x.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)
            weight = torch.view_as_complex(weight.contiguous())
            x = x * weight

        x = torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm='ortho')
        return x


class OriGatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(OriGatedFeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LocalFourierBlock(nn.Module):
    def __init__(self, dim, filer_h, filter_w, dynamic = False, ffn_expansion_factor = 2.66,bias = False):
        super(LocalFourierBlock,self).__init__()
        self.norm1 = LayerNorm(dim)
        self.Filter = LocalFourierFilter(dim,filer_h,filter_w,dynamic)
        self.norm2 = LayerNorm(dim)
        self.ffn = OriGatedFeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x, filter_weight=None):
        x = x + self.Filter(self.norm1(x),filter_weight)
        x = x + self.ffn(self.norm2(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res


class LocalKFourierBlock(nn.Module):
    def __init__(self, dim, filer_h, filter_w, dynamic = False, ffn_expansion_factor = 2.66,bias = False):
        super(LocalKFourierBlock,self).__init__()
        self.norm1 = LayerNorm(dim)
        self.Filter = LocalFourierFilter(dim,filer_h,filter_w,dynamic)
        self.norm2 = LayerNorm(dim)
        self.ffn = ResBlock(dim)
    def forward(self, x, filter_weight=None):
        x = x + self.Filter(self.norm1(x),filter_weight)
        x = x + self.ffn(self.norm2(x))
        return x


class LocalKernelEstimator(nn.Module):
    def __init__(self, dim, filer_h, filter_w, dynamic = False) -> None:
        super(LocalKernelEstimator,self).__init__()
        self.process = LocalKFourierBlock(dim, filer_h, filter_w, dynamic)
        self.GenWeight = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim,dim*2,3,1,1)
        )
    
    def forward(self,input):
        b,c,h,w = input.shape
        out = self.process(input)
        out = self.GenWeight(out)
        return out


class LocalTransFourier(nn.Module):
    def __init__(self,dim,filer_h,filter_w) -> None:
        super(LocalTransFourier,self).__init__()
        self.FilterEstimator = LocalKernelEstimator(dim,filer_h,filter_w,dynamic=False)
        self.filter = LocalFourierBlock(dim,filer_h,filter_w,dynamic=True)
                    
    def forward(self,x):
        weight = self.FilterEstimator(x)
        x = self.filter(x,weight)        
        return x

def block_maker(dim,filer_h,filter_w,type='LocalTrans'):
    if type=='LocalTrans':
        return LocalTransFourier(dim,filer_h,filter_w)

class TransBlock(nn.Module):
    def __init__(self,dim,filer_h,filter_w,n,blcoks,type='LocalTrans') -> None:
        super(TransBlock,self).__init__()
        self.layers = nn.Sequential(*[block_maker(dim,filer_h,filter_w,type) for i in range(blcoks)])
    def forward(self,x):
        x = self.layers(x)        
        return x

class DAFNet(nn.Module):
    def __init__(self,dim=28,stage=2,n_s=[1,1,1],num_blocks=[2,2,2],image_size=256,Multiscale=False,type='LocalTrans',input_fusion=True):
        super(DAFNet,self).__init__()
        self.dim = dim
        self.stage = stage
        self.input_fusion=input_fusion
        if self.input_fusion:
            self.fution = nn.Conv2d(56, 28, 1, 1, 0, bias=False)

        # Input projection
        self.embedding = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)
        
        filter_h = image_size
        filter_w = filter_h // 2 +1

        # Decoder
        self.decoder_layer1 = TransBlock(self.dim*2,filter_h,filter_w,1,2,type)
        self.decoder_layer2 = TransBlock(self.dim,filter_h,filter_w,1,2,type)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                TransBlock(dim_stage,filter_h,filter_w,n_s[i],num_blocks[i],type),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            filter_h = filter_h // 2
            filter_w = filter_h // 2 +1
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = TransBlock(dim_stage,filter_h,filter_w,n_s[-1],num_blocks[-1],type)

        self.ffution1 = nn.Conv2d(self.dim+self.dim*2+self.dim*4, self.dim*2, 1, 1, 0, bias=False)
        self.ffution2 = nn.Conv2d(self.dim*2, self.dim, 1, 1, 0, bias=False)

        
        # Output projection
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)


    def forward(self, x, M=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if M == None:
            M= torch.zeros((1,28,256,256)).cuda()
        if self.input_fusion:
            x = self.fution(torch.cat([x,M],dim=1))
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (layer,DownSample) in self.encoder_layers:
            fea = layer(fea)
            fea_encoder.append(fea) #256 128
            fea = DownSample(fea)

        # Bottleneck
        fea =self.bottleneck(fea) # 64
        f1 = fea_encoder[0]
        f2 = F.interpolate(fea_encoder[1], scale_factor=2)
        f3 = F.interpolate(fea, scale_factor=4)
        f = self.ffution1(torch.cat([f1,f2,f3],dim=1))
        f = self.decoder_layer1(f)
        f = self.ffution2(f)
        f = self.decoder_layer2(f)
        # Decoder


        out = x + self.mapping(f) 
        return out

if __name__=="__main__":
    torch.cuda.set_device(0)
    output = torch.ones([3,28,256,256]).cuda()
    input = output-torch.rand([3,28,256,256]).cuda()
    # input = input.float32()
    mask = torch.rand([3,28,256,256]).cuda()
    model = DAFNet().cuda()
    out = model(input,mask)
    print(0)