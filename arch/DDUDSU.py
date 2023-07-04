import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers.helpers import to_2tuple
import numbers
import scipy.io as scio
import numpy as np


def y2x(Y,step=2):
    [bs, _, row, col] = Y.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step)#.cuda().float()
    for i in range(nC):
        output[:, i, :, :] = Y[:, 0, :, step * i:step * i + col - (nC - 1) * step]
    return output

def x2y(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step)#.cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    output = (torch.sum(output, 1)/nC*2)#.cuda()
    # output = (torch.sum(output, 1)).cuda()
    if len(output.size()) == 3:
        output = output.unsqueeze(1)
    return output

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  
        y = self.conv_du(y)
        return x * y  


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=4, bias=False, act = nn.PReLU()):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

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

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type = 'WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type =='WithBias':
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class DynamicInference(nn.Module):
    def __init__(self,dim,embedding_dim) -> None:
        super(DynamicInference,self).__init__()
        self.GenWeight = nn.Sequential(
            nn.Conv2d(dim*2,embedding_dim,1,1),
            nn.Conv2d(embedding_dim,embedding_dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim,embedding_dim, 3, 1, 1, bias=False),
            nn.Conv2d(embedding_dim,dim,1,1)
        )
        # self.softmax = nn.Softmax(1)
    
    def forward(self,x,y):
        assert(x.shape == y.shape)
        x = self.GenWeight(torch.cat([x,y],dim=1)) + x
        return x
    
class PhiTPhi(nn.Module):
    def __init__(self,dim,embedding_dim) -> None:
        super(PhiTPhi,self).__init__()
        self.GenWeight = nn.Sequential(
            nn.Conv2d(dim,embedding_dim,1,1),
            nn.Conv2d(embedding_dim,embedding_dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim,embedding_dim, 3, 1, 1, bias=False),
            nn.Conv2d(embedding_dim,dim,1,1)
        )
    
    def forward(self,x):
        kernel = self.GenWeight(x) + x
        return kernel

class GradientDescent(nn.Module):
    def __init__(self,dim) -> None:
        super(GradientDescent,self).__init__()
        self.Phi = PhiTPhi(dim,32)
        self.A = DynamicInference(dim,32)
        self.PhiT = PhiTPhi(dim,32)
        self.AT = DynamicInference(dim,32)
        self.Rho = nn.Parameter(torch.Tensor([0.5]))

    def forward(self,x,phi,Y):

        assert(x.shape == phi.shape)

        phi = self.Phi(phi)
        AX = self.A(x,phi)
        phit = self.PhiT(phi)
        res = y2x(x2y(AX)- Y)
        ATres = self.AT(res,phit)  
        x_ = x - self.Rho*ATres

        return x_, phi

class HighChannel(nn.Module):

    def __init__(self, in_channels, hidden_channels, fft_norm='ortho',pool_size=2):

        super(HighChannel, self).__init__()
        
        self.conv_layer1 = torch.nn.Conv2d(in_channels=in_channels * 2 ,
                                          out_channels=hidden_channels * 2,
                                          kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=hidden_channels * 2,
                                          out_channels=in_channels * 2,
                                          kernel_size=1, stride=1, padding=0, bias=False)

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer2(F.gelu(self.conv_layer1(ffted)))  # (batch, c*2, h, w/2+1)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output

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

##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, pool_size=2):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):

        x = self.pool(x)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.uppool(out)
        return out

class mixer(nn.Module):
    def __init__(self,dim, num_heads, bias, pool_size) -> None:
        super().__init__()
        self.low = Attention(dim, num_heads, bias, pool_size)
        self.hi = HighChannel(dim,int(dim*1.5))
        self.expand = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.shirk = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x1, x2 = self.expand(x).chunk(2, dim=1)
        x_hi = self.hi(x1)
        x_low = self.low(x2)
        x = self.shirk(torch.cat([x_hi,x_low],dim=1))
        return x


##  Sparse Transformer Block (STB) 
class STB(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, pool_size=2):
        super(STB, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = mixer(dim, num_heads, bias, pool_size)
        self.norm2 = LayerNorm(dim)
        self.ffn = OriGatedFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class softthr(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(softthr, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = torch.sigmoid(self.project_out(x))
        return x
    
class Denoiser2(nn.Module):
    def __init__(self,in_dim,dim,hidden_dim=32) -> None:
        super(Denoiser2,self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim+hidden_dim,dim,1,1),
            STB(dim,4)
        )
        self.thr = softthr(dim)
        self.proj_forward = nn.Sequential(            
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False)
        )
        self.proj_backward = nn.Sequential(            
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False)
        )

        self.proj_back = nn.Sequential(
            nn.Conv2d(dim*2,dim,1,1),
            STB(dim,4),
            nn.Conv2d(dim,in_dim,1,1)
        )

        self.hidden = nn.Sequential(
            nn.Conv2d(dim*2,hidden_dim,1,1),
            STB(hidden_dim,4)     
        )

    def forward(self,x,hidden_fea,x_ori):
        x_proj = self.proj(torch.cat([x,hidden_fea],dim=1))
        soft_thr = self.thr(x_proj)

        x_fd = self.proj_forward(x_proj)

        x_thr = torch.mul(torch.sign(x_fd), F.relu(torch.abs(x_fd) - soft_thr))

        x_bd = self.proj_backward(x_thr)

        x_proj_back = self.proj_back(torch.cat([x_bd,x_proj],dim=1))

        x_res = self.proj_backward(x_fd) - x_proj

        x_out = x_ori+ x_proj_back
        hidden_fea = self.hidden(torch.cat([x_bd,x_proj],dim=1)) + hidden_fea

        return x_out, x_res, hidden_fea

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ProPorcess(nn.Module):
    def __init__(self, dim=32, expand=2, cs=28):
        super(ProPorcess, self).__init__()
        self.dim = dim
        self.stage = 2
        
        # Input projection
        self.in_proj = nn.Conv2d(cs*2, dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(2):
            self.encoder_layers.append(nn.ModuleList([
                nn.Conv2d(dim_stage, dim_stage * expand, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage * expand, dim_stage * expand, 3, 2, 1, bias=False, groups=dim_stage * expand),
                nn.Conv2d(dim_stage * expand, dim_stage*expand, 1, 1, 0, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = ASPP(dim_stage, [3,6], dim_stage)

        # Decoder:
        self.decoder_layers = nn.ModuleList([])
        for i in range(2):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage // 2, dim_stage, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage, dim_stage, 3, 1, 1, bias=False, groups=dim_stage),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
            ]))
            dim_stage //= 2

        self.out_conv2 = nn.Conv2d(self.dim, cs, 3, 1, 1, bias=False)
        self.fusion1 = nn.Sequential(
            CAB(128),
            nn.Conv2d(128, 16, 3, 1, 1, bias=False)
        )
        self.fusion2 = nn.Sequential(
            CAB(64),
            nn.Conv2d(64, 12, 3, 1, 1, bias=False) 
        )
        self.fusion3 = nn.Sequential(
            CAB(32),
            nn.Conv2d(32, 4, 3, 1, 1, bias=False) 
        )
        self.fusion4 = CAB(32) 
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x,phi):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.lrelu(self.in_proj(torch.cat([x,phi],dim=1)))
        # Encoder
        fea_hi = []
        fea_encoder = []  # [c 2c ]
        for (Conv1, Conv2, Conv3) in self.encoder_layers:
            fea_encoder.append(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
        # Bottleneck
        fea = self.bottleneck(fea)+fea
        fea_hi.append(F.interpolate(self.fusion1(fea), scale_factor=4))
        # Decoder
        for i, (FeaUpSample, Conv1, Conv2, Conv3) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
            fea = fea + fea_encoder[self.stage-1-i]
            if i == 0:
                fea_hi.append(F.interpolate(self.fusion2(fea), scale_factor=2))
            if i == 1:
                fea_hi.append(self.fusion3(fea))
        hidden = self.fusion4(torch.cat(fea_hi, dim=1))
        # Output projection
        out = self.out_conv2(fea)
        return out,hidden


class TransLayer(nn.Module):
    def __init__(self, dim=28):
        super(TransLayer, self).__init__()
        self.mul_conv1 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(dim//2, dim, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(dim//2, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps):
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(feature_maps))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(feature_maps)))
        out = feature_maps * mul + add
        return out

class DegradationUpdate(nn.Module):
    def __init__(self,dim=28) -> None:
        super(DegradationUpdate,self).__init__()
        self.GenDegradationFea = nn.Sequential(
            CAB(dim)
        )
        self.NormLayer = TransLayer(dim)
        self.NormDegradation = TransLayer(dim)
        self.GenDegradationRes = nn.Sequential(
            CAB(dim*2),
            nn.Conv2d(dim*2, dim, 3, 1, 1, bias=False)
        )
        self.Degradation = nn.Sequential(
            nn.Conv2d(dim*2, dim, 3, 1, 1, bias=False)
        )
    
    def forward(self,Degradation_map,meas_HSI,rec_HSI):
        res_hsi = meas_HSI - rec_HSI
        DegradationFea = self.NormLayer(self.GenDegradationFea(meas_HSI))
        DegradationRes = self.GenDegradationRes(torch.cat([DegradationFea,res_hsi],dim=1))
        Degradation = self.NormDegradation(self.Degradation(torch.cat([DegradationRes,Degradation_map],dim=1)))
        return Degradation

class GradientDescentP(nn.Module):
    def __init__(self,dim) -> None:
        super(GradientDescentP,self).__init__()
        self.Phi = PhiTPhi(dim,32)
        self.A = DynamicInference(dim,32)
        self.PhiT = PhiTPhi(dim,32)
        self.AT = DynamicInference(dim,32)
        self.Rho = nn.Parameter(torch.Tensor([0.5]))

    def forward(self,x,degradation,Y):

        assert(x.shape == degradation.shape)

        phi = self.Phi(degradation)
        AX = self.A(x,phi)
        phit = self.PhiT(phi)
        res = y2x(x2y(AX)- Y)
        ATres = self.AT(res,phit) 
        x_ = x - self.Rho*ATres

        return x_

class PhaseP(nn.Module):
    def __init__(self,dim,proj_dim) -> None:
        super(PhaseP,self).__init__()
        self.Degradation = DegradationUpdate(dim) # DSU
        self.GP = GradientDescentP(dim) # DGD
        self.Denoiser = Denoiser2(dim,proj_dim) # DST
    
    def forward(self,x,degradation,Y, meas_HSI, hidden_fea): 
        degradation = self.Degradation(degradation,meas_HSI,x)
        v = self.GP(x,degradation,Y)
        xk, sym_k ,hidden_fea= self.Denoiser(v,hidden_fea,x)
        return xk, sym_k, degradation, hidden_fea

class DDUDSU(nn.Module):
    def __init__(self,dim,stage) -> None:
        super(DDUDSU,self).__init__()
        self.stage = stage
        self.init = ProPorcess(32)
        self.Phases = nn.ModuleList([])
        for i in range(stage):
            self.Phases.append(
                PhaseP(dim,32)
            )
    
    def forward(self,Y,phi=None):
        global gi
        gi = 0
        if len(Y.size()) == 3:
            Y = Y.unsqueeze(1)
        meas_HSI = x = y2x(Y)
        x,hidden_fea = self.init(x,phi)
        degradation = phi
        layers_sym = []
        for phase in self.Phases:
            x, sym_k, degradation, hidden_fea = phase(x, degradation, Y, meas_HSI, hidden_fea) #x, degradation, Y, meas_HSI, hidden_fea
            layers_sym.append(sym_k)
        return x,layers_sym
    

if __name__=="__main__":
    model = DDUDSU(28,7)
    M= torch.randn((1,28,256,256))
    x= torch.randn((1,256,310))
    y = model(x,M)
