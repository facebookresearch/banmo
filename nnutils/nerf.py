import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True, alpha=None):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        if alpha is None:
            self.alpha = self.N_freqs
        else: self.alpha = alpha

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        # consine features
        if self.N_freqs>0:
            shape = x.shape
            bs = shape[0]
            input_dim = shape[-1]
            output_dim = input_dim*(1+self.N_freqs*self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)
            device = x.device

            x = x.view(-1,input_dim)
            out = []
            for freq in self.freq_bands:
                for func in self.funcs:
                    out += [func(freq*x)]
            out =  torch.cat(out, -1)

            ## Apply the window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )
            out = out.view(-1, self.N_freqs, self.nfuncs, input_dim)
            window = self.alpha - torch.arange(self.N_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1,-1, 1, 1)
            out = window * out
            out = out.view(-1,self.N_freqs*self.nfuncs*input_dim)

            out = torch.cat([x, out],-1)
            out = out.view(out_shape)
        else: out = x
        return out



class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, out_channels=3, 
                 skips=[4], raw_feat=False):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        self.weights_reg = []
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i+1}")
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i+1}")
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, out_channels),
                        )

        self.raw_feat = raw_feat

        self.beta = torch.Tensor([1.])
        self.beta = nn.Parameter(self.beta)
        
#        for m in self.modules():
#            if isinstance(m, nn.Linear):
#                if hasattr(m.weight,'data'):
#                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
            raw_feat: does not apply sigmoid

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, 0], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        if self.raw_feat:
            out = rgb
        else:
            rgb = rgb.sigmoid()
            out = torch.cat([rgb, sigma], -1)
        return out


class RTHead(NeRF):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, is_bone, use_cam, **kwargs):
        super(RTHead, self).__init__(**kwargs)
        self.is_bone=is_bone
        self.use_cam=use_cam

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        x = super(RTHead, self).forward(x)
        rts = x.view(-1,7) 

        rquat=rts[:,:4]
        if self.is_bone or self.use_cam:
            rquat[:,0]+=10
        rquat=F.normalize(rquat,2,-1)

        tmat= rts[:,4:7] *0.1

        rts = torch.cat([rquat,tmat],-1)
        x = rts.view(x.shape)
        return x
    

def evaluate_mlp(model, embedded, chunk, code=None, sigma_only=False):
    B,nbins,_ = embedded.shape
    out_chunks = []
    for i in range(0, B, chunk):
        if code is not None:
            embedded = torch.cat([embedded[i:i+chunk],
                       code[i:i+chunk].repeat(1,nbins,1)], -1)
        out_chunks += [model(embedded, sigma_only=sigma_only)]

    out = torch.cat(out_chunks, 0)
    return out
