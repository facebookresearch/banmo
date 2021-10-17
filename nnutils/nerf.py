import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from pytorch3d import transforms
import trimesh

import sys
sys.path.insert(0, 'third_party')
from ext_nnutils.net_blocks import conv2d, net_init, fc_stack
from ext_nnutils.geom_utils import R_2vect

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
                 in_channels_xyz=63, in_channels_dir=27,
                 out_channels=3, 
                 skips=[4], raw_feat=False, init_beta=1./100, activation=nn.ReLU(True)):
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
        self.use_xyz = False

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
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                activation)

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, out_channels),
                        )

        self.raw_feat = raw_feat

        self.beta = torch.Tensor([init_beta])
        self.beta = nn.Parameter(self.beta)
        
#        for m in self.modules():
#            if isinstance(m, nn.Linear):
#                if hasattr(m.weight,'data'):
#                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x ,xyz=None, sigma_only=False):
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

class Transhead(NeRF):
    """
    translation head
    """
    def __init__(self, **kwargs):
        super(Transhead, self).__init__(**kwargs)

    def forward(self, x, xyz=None,sigma_only=False):
        flow = super(Transhead, self).forward(x, sigma_only=sigma_only)
        flow = flow*0.1
        return flow

class SE3head(NeRF):
    """
    modify the output to be rigid transforms per point
    modified from Nerfies
    """
    def __init__(self, **kwargs):
        super(SE3head, self).__init__(**kwargs)
        self.use_xyz=True

    def forward(self, x, xyz=None,sigma_only=False):
        x = super(SE3head, self).forward(x, sigma_only=sigma_only)
        x = x.view(-1,9)
        rotation, pivot, translation = x.split([3,3,3],-1)
        pivot = pivot*0.1
        translation = translation*0.1
        
        shape = xyz.shape
        warped_points = xyz.view(-1,3).clone()
        warped_points = warped_points + pivot
        rotmat = transforms.so3_exponential_map(rotation)
        warped_points = rotmat.matmul(warped_points[...,None])[...,0]
        warped_points = warped_points - pivot
        warped_points = warped_points + translation

        flow = warped_points.view(shape) - xyz
        return flow

class RTHead(NeRF):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, use_quat, **kwargs):
        super(RTHead, self).__init__(**kwargs)
        # use quaternion when estimating full rotation
        # use exponential map when estimating delta rotation
        self.use_quat=use_quat
        if self.use_quat: self.num_output=7
        else: self.num_output=6

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(RTHead, self).forward(x)
        bs = x.shape[0]
        rts = x.view(-1,self.num_output)  # bs B,x
        B = rts.shape[0]//bs

        tmat= rts[:,0:3] *0.1

        if self.use_quat:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        else:
            #TODO add more variance to the network
            #rot=rts[:,3:6] * 10
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts
    
class RTExplicit(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, delta=False):
        super(RTExplicit, self).__init__()
        self.max_t = max_t
        self.delta = delta

        # initialize rotation
        trans = torch.zeros(max_t, 3)
        if delta:
            rot = torch.zeros(max_t, 3) 
        else:
            rot = torch.rand(max_t, 4) * 2 - 1
        se3 = torch.cat([trans, rot],-1)

        self.se3 = nn.Parameter(se3)
        self.num_output = se3.shape[-1]


    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        bs = x.shape[0]
        x = self.se3[x] # bs B,x
        rts = x.view(-1,self.num_output)
        B = rts.shape[0]//bs
        
        tmat= rts[:,0:3] *0.1

        if self.delta:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        else:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts

class ScoreHead(NeRF):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, recursion_level, **kwargs):
        super(ScoreHead, self).__init__(**kwargs)
        grid= generate_healpix_grid(recursion_level=recursion_level)
        self.register_buffer('grid', grid)
        self.num_scores = self.grid.shape[0]

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(ScoreHead, self).forward(x)
        bs = x.shape[0]
        x = x.view(-1,self.num_scores+3)  # bs B,x

        #TODO do not use tmat since it is not trained
        tmat = x[:,0:3]*0.
        scores=x[:,3:]
        if self.training:
            return scores, self.grid
        else:
            scores = scores.view(bs,-1,1)
            rmat = self.grid[None].repeat(bs,1,1,1)
            tmat = tmat[:,None].repeat(1,self.num_scores,1)
            rmat = rmat.view(bs,-1,9)
            rts = torch.cat([scores,rmat, tmat],-1)
            rts = rts.view(bs,self.num_scores,-1)
            return rts

class ResNetConv(nn.Module):
    def __init__(self, in_channels):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        if in_channels!=3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc=None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, in_channels=3,out_channels=128, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(in_channels=in_channels)
        self.conv1 = conv2d(batch_norm, 512, 128, stride=1, kernel_size=3)
        net_init(self.conv1)

    def forward(self, img):
        feat = self.resnet_conv.forward(img) # 512,4,4
        feat = self.conv1(feat) # 128,4,4
        feat = F.max_pool2d(feat, 4, 4)
        feat = feat.view(img.size(0), -1)
        return feat

def evaluate_mlp(model, xyz_embedded, embed_xyz=None, dir_embedded=None,
                chunk=32*1024, 
                xyz=None,
                code=None, sigma_only=False):
    """
    embed_xyz: embedding function
    chunk is the point-level chunk divided by number of bins
    """
    B,nbins,_ = xyz_embedded.shape
    out_chunks = []
    for i in range(0, B, chunk):
        embedded = xyz_embedded[i:i+chunk]
        if embed_xyz is not None:
            embedded = embed_xyz(embedded)
        if dir_embedded is not None:
            embedded = torch.cat([embedded,
                       dir_embedded[i:i+chunk]], -1)
        if code is not None:
            code_chunk = code[i:i+chunk]
            if code_chunk.dim() == 2: 
                code_chunk = code_chunk[:,None]
            code_chunk = code_chunk.repeat(1,nbins,1)
            embedded = torch.cat([embedded,code_chunk], -1)
        if xyz is not None:
            xyz_chunk = xyz[i:i+chunk]
        else: xyz_chunk = None
        out_chunks += [model(embedded, sigma_only=sigma_only, xyz=xyz_chunk)]

    out = torch.cat(out_chunks, 0)
    return out

def generate_healpix_grid(recursion_level=None, size=None):
    """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).
    Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
    along the 'tilt' direction 6*2**recursion_level times over 2pi.
    Args:
      recursion_level: An integer which determines the level of resolution of the
        grid.  The final number of points will be 72*8**recursion_level.  A
        recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
        for evaluation.
      size: A number of rotations to be included in the grid.  The nearest grid
        size in log space is returned.
    Returns:
      (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
      taken from implicit pdf https://implicit-pdf.github.io/
    """
    import healpy as hp  # pylint: disable=g-import-not-at-top

    assert not(recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size/72.)/np.log(8.))), 0)
    number_per_side = 2**recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    # Take these points on the sphere and
    unit_vec = np.asarray([0,0,1])
    rot_mat = torch.Tensor([ R_2vect(unit_vec, i) for i in s2_points])

    rot_mats = []
    tilts = np.linspace(0, 2*np.pi, 6*2**recursion_level, endpoint=False)
    tilts = torch.Tensor(tilts)
    for tilt in tilts:
        zrot_mat = transforms.axis_angle_to_matrix(torch.Tensor([[0., 0., tilt]]))
        rot_mats.append(rot_mat.matmul(zrot_mat))
  
    rot_mats = torch.cat(rot_mats, 0)
    #vecs = rot_mats.matmul(torch.Tensor([0,0,1])[None,:,None])[...,0]
    #trimesh.Trimesh(vecs).export('0.obj')
    return rot_mats
