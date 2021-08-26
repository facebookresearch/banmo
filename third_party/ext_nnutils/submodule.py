# MIT License
# 
# Copyright (c) 2019 Carnegie Mellon University
# Copyright (c) 2021 Google LLC
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# modified from https://github.com/gengshan-y/VCN

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb
import kornia

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1,with_bn=True):
        super(residualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1

        if with_bn:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1)
        else:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation,with_bn=False)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, with_bn=False)
        self.downsample = downsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True))


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        bias = not with_bn

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()
        bias = not with_bn
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, with_bn=True, levels=4):
        super(pyramidPooling, self).__init__()
        self.levels = levels

        self.paths = []
        for i in range(levels):
            self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        for pool_size in np.linspace(1,min(h,w)//2,self.levels,dtype=int):
            k_sizes.append((int(h/pool_size), int(w/pool_size)))
            strides.append((int(h/pool_size), int(w/pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]

        pp_sum = x

        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            pp_sum = pp_sum + 1./self.levels*out
        pp_sum = self.relu(pp_sum/2.)

        return pp_sum

class pspnet(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """
    def __init__(self, is_proj=True,groups=1):
        super(pspnet, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)

        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1))
        self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                 padding=1, stride=1)

        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)
            self.proj2 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.upsample(conv6, [conv5.size()[2],conv5.size()[3]],mode='bilinear')
        concat5 = torch.cat((conv5,self.upconv6[1](conv6x)),dim=1)
        conv5 = self.iconv5(concat5) 

        conv5x = F.upsample(conv5, [conv4.size()[2],conv4.size()[3]],mode='bilinear')
        concat4 = torch.cat((conv4,self.upconv5[1](conv5x)),dim=1)
        conv4 = self.iconv4(concat4) 

        conv4x = F.upsample(conv4, [rconv3.size()[2],rconv3.size()[3]],mode='bilinear')
        concat3 = torch.cat((rconv3,self.upconv4[1](conv4x)),dim=1)
        conv3 = self.iconv3(concat3) 

        conv3x = F.upsample(conv3, [pool1.size()[2],pool1.size()[3]],mode='bilinear')
        concat2 = torch.cat((pool1,self.upconv3[1](conv3x)),dim=1)
        conv2 = self.iconv2(concat2) 

        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            proj2 = self.proj2(conv2)
            return proj6,proj5,proj4,proj3,proj2
        else:
            return conv6, conv5, conv4, conv3, conv2


class pspnet_s(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """
    def __init__(self, is_proj=True,groups=1):
        super(pspnet_s, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)

        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        #self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
        #                             conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
        #                                         padding=1, stride=1))
        #self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
        #                                         padding=1, stride=1)

        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128,k_size=1,n_filters=128//groups, padding=0,stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)
            #self.proj2 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.upsample(conv6, [conv5.size()[2],conv5.size()[3]],mode='bilinear')
        concat5 = torch.cat((conv5,self.upconv6[1](conv6x)),dim=1)
        conv5 = self.iconv5(concat5) 

        conv5x = F.upsample(conv5, [conv4.size()[2],conv4.size()[3]],mode='bilinear')
        concat4 = torch.cat((conv4,self.upconv5[1](conv5x)),dim=1)
        conv4 = self.iconv4(concat4) 

        conv4x = F.upsample(conv4, [rconv3.size()[2],rconv3.size()[3]],mode='bilinear')
        concat3 = torch.cat((rconv3,self.upconv4[1](conv4x)),dim=1)
        conv3 = self.iconv3(concat3) 

        #conv3x = F.upsample(conv3, [pool1.size()[2],pool1.size()[3]],mode='bilinear')
        #concat2 = torch.cat((pool1,self.upconv3[1](conv3x)),dim=1)
        #conv2 = self.iconv2(concat2) 

        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
        #    proj2 = self.proj2(conv2)
        #    return proj6,proj5,proj4,proj3,proj2
            return proj6,proj5,proj4,proj3
        else:
        #    return conv6, conv5, conv4, conv3, conv2
            return conv6, conv5, conv4, conv3

class bfmodule(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(bfmodule, self).__init__()
        self.proj = conv2DBatchNormRelu(in_channels=inplanes,k_size=1,n_filters=64,padding=0,stride=1)
        self.inplanes = 64
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)
        # Iconvs
        self.upconv6 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.upconv5 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        self.upconv4 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        self.upconv3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        self.iconv5 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.iconv4 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.iconv3 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.iconv2 = nn.Sequential(conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                                 padding=1, stride=1),
                                    nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True))

        self.proj6 = nn.Conv2d(128, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
        self.proj5 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
        self.proj4 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
        self.proj3 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        proj = self.proj(x) # 4x
        rconv3 = self.res_block3(proj) #8x
        conv4 = self.res_block5(rconv3) #16x
        conv5 = self.res_block6(conv4) #32x
        conv6 = self.res_block7(conv5) #64x
        conv6 = self.pyramid_pooling(conv6) #64x
        pred6 = self.proj6(conv6)

        conv6u = F.upsample(conv6, [conv5.size()[2],conv5.size()[3]], mode='bilinear')
        concat5 = torch.cat((conv5,self.upconv6(conv6u)),dim=1) 
        conv5 = self.iconv5(concat5) #32x
        pred5 = self.proj5(conv5)

        conv5u = F.upsample(conv5, [conv4.size()[2],conv4.size()[3]], mode='bilinear')
        concat4 = torch.cat((conv4,self.upconv5(conv5u)),dim=1)
        conv4 = self.iconv4(concat4) #16x
        pred4 = self.proj4(conv4)

        conv4u = F.upsample(conv4, [rconv3.size()[2],rconv3.size()[3]], mode='bilinear')
        concat3 = torch.cat((rconv3,self.upconv4(conv4u)),dim=1)
        conv3 = self.iconv3(concat3) # 8x
        pred3 = self.proj3(conv3)

        conv3u = F.upsample(conv3, [x.size()[2],x.size()[3]], mode='bilinear')
        concat2 = torch.cat((proj,self.upconv3(conv3u)),dim=1)
        pred2 = self.iconv2(concat2)  # 4x

        return pred2, pred3, pred4, pred5, pred6

class bfmodule_feat(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(bfmodule_feat, self).__init__()
        self.proj = conv2DBatchNormRelu(in_channels=inplanes,k_size=1,n_filters=64,padding=0,stride=1)
        self.inplanes = 64
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)
        # Iconvs
        self.upconv6 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.upconv5 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        self.upconv4 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        self.upconv3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        self.iconv5 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.iconv4 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.iconv3 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.iconv2 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                                 padding=1, stride=1)

        self.proj6 = nn.Conv2d(128, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
        self.proj5 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
        self.proj4 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
        self.proj3 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)
        self.proj2 = nn.Conv2d(64, outplanes,kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        proj = self.proj(x) # 4x
        rconv3 = self.res_block3(proj) #8x
        conv4 = self.res_block5(rconv3) #16x
        conv5 = self.res_block6(conv4) #32x
        conv6 = self.res_block7(conv5) #64x
        conv6 = self.pyramid_pooling(conv6) #64x
        pred6 = self.proj6(conv6)

        conv6u = F.upsample(conv6, [conv5.size()[2],conv5.size()[3]], mode='bilinear')
        concat5 = torch.cat((conv5,self.upconv6(conv6u)),dim=1) 
        conv5 = self.iconv5(concat5) #32x
        pred5 = self.proj5(conv5)

        conv5u = F.upsample(conv5, [conv4.size()[2],conv4.size()[3]], mode='bilinear')
        concat4 = torch.cat((conv4,self.upconv5(conv5u)),dim=1)
        conv4 = self.iconv4(concat4) #16x
        pred4 = self.proj4(conv4)

        conv4u = F.upsample(conv4, [rconv3.size()[2],rconv3.size()[3]], mode='bilinear')
        concat3 = torch.cat((rconv3,self.upconv4(conv4u)),dim=1)
        conv3 = self.iconv3(concat3) # 8x
        pred3 = self.proj3(conv3)

        conv3u = F.upsample(conv3, [x.size()[2],x.size()[3]], mode='bilinear')
        concat2 = torch.cat((proj,self.upconv3(conv3u)),dim=1)
        conv2 = self.iconv2(concat2)  # 4x
        pred2 = self.proj2(conv2)  # 4x
        return pred2, conv2


def compute_geo_costs(rot, trans, Ex, Kinv, hp0, hp1, tau, Kinv_n=None):
    if Kinv_n is None: Kinv_n = Kinv
    R01 = kornia.angle_axis_to_rotation_matrix(rot)
    H01 = Kinv.inverse().matmul(R01).matmul(Kinv_n)
    comp_hp1 = H01.matmul(hp1.permute(0,2,1))
    foe = (comp_hp1-tau*hp0.permute(0,2,1))
    parallax3d = Kinv.matmul(foe)
    p3dmag = parallax3d.norm(2,1)[:,np.newaxis]
    parallax2d = (comp_hp1/comp_hp1[:,-1:]-hp0.permute(0,2,1))[:,:2]
    p2dmag = parallax2d.norm(2,1)[:,np.newaxis]
    p2dnorm = parallax2d / (1e-9+p2dmag)
    foe_cam = Kinv.inverse().matmul(trans[:,:,np.newaxis])
    foe_cam = foe_cam[:,:2] / (1e-9+foe_cam[:,-1:])
    direct = foe_cam -hp0.permute(0,2,1)[:,:2]
    directn = direct / (1e-9+direct.norm(2,1)[:,np.newaxis])

    # metrics: 0) R-homography+symterr; 1) sampson 2) 2D angular 3) 3D sampson 4) 3D angular
    ##TODO validate
    comp_hp0 = H01.inverse().matmul(hp0.permute(0,2,1))
    mcost00 = parallax2d.norm(2,1)
    mcost01 = (comp_hp0/comp_hp0[:,-1:] - hp1.permute(0,2,1))[:,:2].norm(2,1)
    mcost1 = sampson_err(Kinv.matmul(hp0.permute(0,2,1)),
                         Kinv_n.matmul(hp1.permute(0,2,1)),Ex.cuda().permute(0,2,1))  # variable K
    mcost2 = -(trans[:,-1:,np.newaxis]).sign()*(directn*p2dnorm).sum(1,keepdims=True)
    mcost4 = -(trans[:,:,np.newaxis]*parallax3d).sum(1,keepdims=True)/(p3dmag+1e-9)
    mcost3 = torch.clamp(1-mcost4.pow(2),0,1).sqrt()*p3dmag*mcost4.sign()
    mcost10 = torch.clamp(1-mcost2.pow(2),0,1).sqrt()*p2dmag*mcost2.sign()
    return mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag, mcost10

def get_skew_mat(transx,rotx):
    rot = kornia.angle_axis_to_rotation_matrix(rotx)
    trans = -rot.permute(0,2,1).matmul(transx[:,:,np.newaxis])[:,:,0]
    rot = rot.permute(0,2,1)
    tx = torch.zeros(transx.shape[0],3,3)
    tx[:,0,1] = -transx[:,2]
    tx[:,0,2] = transx[:,1]
    tx[:,1,0] = transx[:,2]
    tx[:,1,2] = -transx[:,0]
    tx[:,2,0] = -transx[:,1]
    tx[:,2,1] = transx[:,0]
    return rot.matmul(tx)

def sampson_err(x1h, x2h, F):
    l2 = F.permute(0,2,1).matmul(x1h)
    l1 = F.matmul(x2h)
    algdis = (l1 * x1h).sum(1)
    dis = algdis**2 /  (1e-9+l1[:,0]**2+l1[:,1]**2+l2[:,0]**2+l2[:,1]**2)
    return dis


def get_intrinsics(intr, noise=False):
    f =  intr[0].float()
    cx = intr[1].float()
    cy = intr[2].float()
    bs = f.shape[0]

    delta = 1e-4
    if noise:
        fo = f.clone()
        cxo = cx.clone()
        cyo = cy.clone()
        f = torch.Tensor(np.random.normal(loc=0., scale=delta,size=(bs,))).cuda().exp() * fo
        cx = torch.Tensor(np.random.normal(loc=0.,scale=delta,size=(bs,))).cuda().exp() * cxo
        cy = torch.Tensor(np.random.normal(loc=0.,scale=delta,size=(bs,))).cuda().exp() * cyo

    Kinv = torch.Tensor(np.eye(3)[np.newaxis]).cuda().repeat(bs,1,1)
    Kinv[:,2,2] *= f
    Kinv[:,0,2] -= cx
    Kinv[:,1,2] -= cy
    Kinv /= f[:,np.newaxis,np.newaxis] #4,3,3

    Taug = torch.cat(intr[4:10],-1).view(-1,bs).T # 4,6
    Taug = torch.cat((Taug.view(bs,3,2).permute(0,2,1),Kinv[:,2:3]),1)
    Kinv = Kinv.matmul(Taug)
    if len(intr)>12:
        Kinv_n = torch.Tensor(np.eye(3)[np.newaxis]).cuda().repeat(bs,1,1)
        fn = intr[12].float()
        Kinv_n[:,2,2] *= fn
        Kinv_n[:,0,2] -= cx
        Kinv_n[:,1,2] -= cy
        Kinv_n /= fn[:,np.newaxis,np.newaxis] #4,3,3
    elif noise:
        f = torch.Tensor(np.random.normal(loc=0., scale=delta,size=(bs,))).cuda().exp() * fo
        cx = torch.Tensor(np.random.normal(loc=0.,scale=delta,size=(bs,))).cuda().exp() * cxo
        cy = torch.Tensor(np.random.normal(loc=0.,scale=delta,size=(bs,))).cuda().exp() * cyo

        Kinv_n = torch.Tensor(np.eye(3)[np.newaxis]).cuda().repeat(bs,1,1)
        Kinv_n[:,2,2] *= f
        Kinv_n[:,0,2] -= cx
        Kinv_n[:,1,2] -= cy
        Kinv_n /= f[:,np.newaxis,np.newaxis] #4,3,3

        Taug = torch.cat(intr[4:10],-1).view(-1,bs).T # 4,6
        Taug = torch.cat((Taug.view(bs,3,2).permute(0,2,1),Kinv_n[:,2:3]),1)
        Kinv_n = Kinv_n.matmul(Taug)
    else:
        Kinv_n = Kinv

    return Kinv, Kinv_n

def F_ngransac(hp0,hp1,Ks,rand, unc_occ, iters=1000,cv=False,Kn=None):
    if Kn is None:
        Kn = Ks
    import cv2

    b = hp1.shape[0]
    hp0_cpu = np.asarray(hp0.cpu())
    hp1_cpu = np.asarray(hp1.cpu())
    if not rand:
        ## TODO
        fmask = np.ones(hp0.shape[1]).astype(bool)
        rand_seed = 0
    else:
        fmask = np.random.choice([True, False], size=hp0.shape[1], p=[0.1,0.9])
        rand_seed = np.random.randint(0,1000) # random seed to by used in C++
    ### TODO
    hp0 = Ks.inverse().matmul(hp0.permute(0,2,1)).permute(0,2,1)
    hp1 = Kn.inverse().matmul(hp1.permute(0,2,1)).permute(0,2,1)
    ratios = torch.zeros(hp0[:1,:,:1].shape)
    probs = torch.Tensor(np.ones(fmask.sum()))/fmask.sum()
    probs = probs[np.newaxis,:,np.newaxis]

    #probs = torch.Tensor(np.zeros(fmask.sum()))
    ##unc_occ = unc_occ<0; probs[unc_occ[0]] = 1./unc_occ.float().sum()
    #probs = F.softmax(-0.1*unc_occ[0],-1).cpu()
    #probs = probs[np.newaxis,:,np.newaxis]

    Es = torch.zeros((b, 3,3)).float() # estimated model
    rot = torch.zeros((b, 3)).float() # estimated model
    trans = torch.zeros((b, 3)).float() # estimated model
    out_model = torch.zeros((3, 3)).float() # estimated model
    out_inliers = torch.zeros(probs.size()) # inlier mask of estimated model
    out_gradients = torch.zeros(probs.size()) # gradient tensor (only used during training)

    for i in range(b):
        pts1 = hp0[i:i+1, fmask,:2].cpu()
        pts2 = hp1[i:i+1, fmask,:2].cpu()
        # create data tensor of feature coordinates and matching ratios
        correspondences = torch.cat((pts1, pts2, ratios), axis=2)
        correspondences = correspondences.permute(2,1,0)
        #incount = ngransac.find_fundamental_mat(correspondences, probs, rand_seed, 1000, 0.1, True, out_model, out_inliers, out_gradients)
        #E = K1.T.dot(out_model).dot(K0)

        if cv==True:
            E, ffmask = cv2.findEssentialMat(np.asarray(pts1[0]), np.asarray(pts2[0]), np.eye(3), cv2.FM_RANSAC,threshold=0.0001)
            ffmask = ffmask[:,0]
            Es[i]=torch.Tensor(E)
        else:
            import ngransac
            incount = ngransac.find_essential_mat(correspondences, probs, rand_seed, iters, 0.0001, out_model, out_inliers, out_gradients)
            Es[i]=out_model
            E = np.asarray(out_model)
            maskk = np.asarray(out_inliers[0,:,0])
            ffmask = fmask.copy()
            ffmask[fmask] = maskk
        K1 = np.asarray(Kn[i].cpu())
        K0 = np.asarray(Ks[i].cpu())
        R1, R2, T = cv2.decomposeEssentialMat(E)
        for rott in [(R1,T),(R2,T),(R1,-T),(R2,-T)]:
            if testEss(K0,K1,rott[0],rott[1],hp0_cpu[0,ffmask].T, hp1_cpu[i,ffmask].T):
            #if testEss(K0,K1,rott[0],rott[1],hp0_cpu[0,ffmask].T[:,ffmask.sum()//10::ffmask.sum()//10], hp1_cpu[i,ffmask].T[:,ffmask.sum()//10::ffmask.sum()//10]):
                R01=rott[0].T
                t10=-R01.dot(rott[1][:,0])
        if not 't10' in locals():
            t10 = np.asarray([0,0,1])
            R01 = np.eye(3)
        rot[i] = torch.Tensor(cv2.Rodrigues(R01)[0][:,0]).cuda()
        trans[i] = torch.Tensor(t10).cuda()

    return rot, trans, Es


def testEss(K0,K1,R,T,p1,p2):
    import cv2
    testP = cv2.triangulatePoints(K0.dot(np.concatenate( (np.eye(3),np.zeros((3,1))), -1)), 
                          K1.dot(np.concatenate( (R,T), -1)), 
                          p1[:2],p2[:2])
    Z1 = testP[2,:]/testP[-1,:]
    Z2 = (R.dot(Z1*np.linalg.inv(K0).dot(p1))+T)[-1,:]
    if ((Z1>0).sum() > (Z1<=0).sum()) and ((Z2>0).sum() > (Z2<=0).sum()):
        #print(Z1)
        #print(Z2)
        return True
    else:
        return False
