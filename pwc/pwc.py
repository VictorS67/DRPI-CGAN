import getopt
import math
import numpy as np
import PIL
import sys
import torch

from PIL import Image
from torch import nn

try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.leakyRELU = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv1b = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            self.conv1a, self.leakyRELU, 
            self.conv1b, self.leakyRELU, 
            self.conv1b, self.leakyRELU
        )

        self.conv2a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            self.conv2a, self.leakyRELU, 
            self.conv2b, self.leakyRELU, 
            self.conv2b, self.leakyRELU
        )

        self.conv3a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Sequential(
            self.conv3a, self.leakyRELU, 
            self.conv3b, self.leakyRELU, 
            self.conv3b, self.leakyRELU
        )

        self.conv4a = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1)
        self.conv4b = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Sequential(
            self.conv4a, self.leakyRELU, 
            self.conv4b, self.leakyRELU, 
            self.conv4b, self.leakyRELU
        )

        self.conv5a = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv5b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Sequential(
            self.conv5a, self.leakyRELU, 
            self.conv5b, self.leakyRELU, 
            self.conv5b, self.leakyRELU
        )

        self.conv6a = nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1)
        self.conv6b = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Sequential(
            self.conv6a, self.leakyRELU, 
            self.conv6b, self.leakyRELU, 
            self.conv6b, self.leakyRELU
        )

    def forward(self, x):
        flow1 = self.conv1(x)
        flow2 = self.conv2(flow1)
        flow3 = self.conv3(flow2)
        flow4 = self.conv4(flow3)
        flow5 = self.conv5(flow4)
        flow6 = self.conv6(flow5)

        return [flow1, flow2, flow3, flow4, flow5, flow6]


class Decoder(nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
        intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

        if intLevel < 6:
            self.netUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
            self.netUpfeat = nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
            self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

        self.leakyRELU = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        self.dc_conv1a = nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dc_conv1 = nn.Sequential(self.dc_conv1a, self.leakyRELU)

        self.dc_conv2a = nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dc_conv2 = nn.Sequential(self.dc_conv2a, self.leakyRELU)

        self.dc_conv3a = nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.dc_conv3 = nn.Sequential(self.dc_conv3a, self.leakyRELU)

        self.dc_conv4a = nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dc_conv4 = nn.Sequential(self.dc_conv4a, self.leakyRELU)

        self.dc_conv5a = nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dc_conv5 = nn.Sequential(self.dc_conv5a, self.leakyRELU)

        self.dc_conv6a = nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.dc_conv6 = nn.Sequential(self.dc_conv6a)

    def forward(self, tenOne, tenTwo, objPrevious):
        tenFlow = None
        tenFeat = None

        if objPrevious is None:
            tenFlow = None
            tenFeat = None

            tenVolume = nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)
            tenFeat = torch.cat([tenVolume], 1)
        else:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])

            tenVolume = nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)
            tenFeat = torch.cat([tenVolume, tenOne, tenFlow, tenFeat], 1)

        tenFeat = torch.cat([self.dc_conv1(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.dc_conv2(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.dc_conv3(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.dc_conv4(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.dc_conv5(tenFeat), tenFeat], 1)

        tenFlow = self.dc_conv6(tenFeat)

        return {'tenFlow': tenFlow, 'tenFeat': tenFeat}


class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.leakyRELU = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        self.rf_conv1 = nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.rf_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.rf_conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.rf_conv4 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.rf_conv5 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.rf_conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.rf_conv7 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)

        self.refiner = nn.Sequential(
            self.rf_conv1, self.leakyRELU, 
            self.rf_conv2, self.leakyRELU, 
            self.rf_conv3, self.leakyRELU,
            self.rf_conv4, self.leakyRELU,
            self.rf_conv5, self.leakyRELU,
            self.rf_conv6, self.leakyRELU,
            self.rf_conv7
        )

    def forward(self, x):
        return self.refiner(x)


class Pwc(nn.Module):
    def __init__(self):
        super(Pwc, self).__init__()

        self.extractor = Extractor()

        self.decoder1 = Decoder(2)
        self.decoder2 = Decoder(3)
        self.decoder3 = Decoder(4)
        self.decoder4 = Decoder(5)
        self.decoder5 = Decoder(6)

        self.refiner = Refiner()

    def forward(self, img1, img2):
        tenOne = self.netExtractor(img1)
        tenTwo = self.netExtractor(img2)

        objEstimate = self.decoder5(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.decoder4(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.decoder3(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.decoder2(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.decoder1(tenOne[-5], tenTwo[-5], objEstimate)

        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0


def pwc_net(arguments_strModel='default'): # 'default', or 'chairs-things'
    model = Pwc()
    model.load_state_dict({
        strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items()
    })

    return model
