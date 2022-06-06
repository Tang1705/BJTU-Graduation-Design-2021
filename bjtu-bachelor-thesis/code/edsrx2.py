import os
import torch.nn as nn
import torch.optim as optim
from base_networks2 import *
from generate_grad import *
from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        #### 
        elif scale_factor == 16:
          kernel = 20
          stride = 16
          padding = 2
        
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)


        self.color2gard = Gradient_Map()

        self.featc0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.featc1 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.featg2 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.feat2_0 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat2_1 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)


        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(2*base_filter, base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = DownBlock(2*base_filter, base_filter, kernel, stride, padding)
        self.up3 = UpBlock(base_filter, kernel, stride, padding)
        self.down3 = DownBlock(2*base_filter, base_filter, kernel, stride, padding)
        self.up4 = UpBlock(base_filter, kernel, stride, padding)

        self.down4 = DownBlock(base_filter, base_filter, kernel, stride, padding)
        self.up5 = UpBlock(base_filter, kernel, stride, padding)
        self.down5 = DownBlock(base_filter, base_filter, kernel, stride, padding)
        self.up6 = UpBlock(base_filter, kernel, stride, padding)
        self.down6 = DownBlock(base_filter, base_filter, kernel, stride, padding)
        self.up7 = UpBlock(base_filter, kernel, stride, padding)
        self.down7 = DownBlock(base_filter, base_filter, kernel, stride, padding)
        self.up8 = UpBlock(base_filter, kernel, stride, padding)



        self.Gblock1 = ResnetBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
        self.Gblock2 = ResnetBlock(2*base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
        self.Gblock3 = ResnetBlock(2*base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
        self.Gblock4 = ResnetBlock(2*base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
        self.Gblock5 = ResnetBlock(2*base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)

        
        self.out_depth1 = ConvBlock(2*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.out_depth2 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_depth_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        self.out_SR1 = ConvBlock(2*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.out_SR2 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_SR_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        self.out_SR_f = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.out_SR2 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_SR_conv_f = ConvBlock(base_filter, num_channels, 1, 1, 0, activation='prelu', norm=None)

        self.input_lr1 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.Gblock6 = ResnetBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
        self.Gblock7 = ResnetBlock(base_filter, base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
        self.input_lr2 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.input_lr = ConvBlock(base_filter, num_channels, 1, 1, 0, activation=None, norm=None)


      
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, x, color):

        gard = self.color2gard(color)
        # print(x.shape, 'xxxxxxxxxx')
        # print(color.shape, 'ccccccccccccccc')

        x1 = self.feat0(x)
        x2 = self.feat1(x1)

        c1 = self.featc0(gard)
        c2 = self.featc1(c1)


        h1 = self.up1(x2)
        c_res1 = self.Gblock1(c2)
        concat_1 = torch.cat((h1, c_res1), 1)

        l1 = self.down1(concat_1)
        h2 = self.up2(l1)
        c_res2 = self.Gblock2(concat_1)
        concat_2 = torch.cat((h2, c_res2), 1)

        l2 = self.down2(concat_2)
        h3 = self.up3(l2)
        c_res3 = self.Gblock3(concat_2)
        concat_3 = torch.cat((h3, c_res3), 1)

        l3 = self.down3(concat_3)
        h4 = self.up4(l3)
        c_res4 = self.Gblock4(concat_3)
        concat_4 = torch.cat((h4, c_res4), 1)

        # l4 = self.down4(concat_4)
        # h5 = self.up5(l4)
        # c_res5 = self.Gblock5(concat_4)
        # concat_5 = torch.cat((h5, c_res5), 1)

        d1 = self.out_depth1(concat_4)
        # d2 = self.out_depth2(d1)
        d = self.output_depth_conv(d1)

        sr1 = self.out_SR1(concat_4)
        sr = self.output_SR_conv(sr1)

        # input_lr1 = self.down_input1(sr1)
        # input_lr2 = self.down_input2(input_lr1)
        # grad1 = self.color2gard(d)
        # grad2 = self.color2gard(sr)
        # gt_grad = self.color2gard(target)

###########step2
        x21 = self.feat2_0(d1)
        x22 = self.feat2_1(x21)
        
        l4 = self.down4(x22)
        h5 = self.up5(l4)

        l5 = self.down5(h5)
        h6 = self.up6(l5)

        l6 = self.down6(h6)
        h7 = self.up7(l6)

        l7 = self.down7(h7)
        h8 = self.up8(l7)

        concat_all = torch.cat((h5, h6, h7, h8), 1)

        out1 = self.out_SR_f(concat_all)
        out = self.output_SR_conv_f(out1)

        lr1 = self.input_lr1(out)
        lr2 = self.Gblock6(lr1)
        lr3 = self.Gblock7(lr2)
        lr2 = self.input_lr2(lr3)
        lr = self.input_lr(lr2)

        return d, sr, out, lr#grad1, grad2, gt_grad#, input_lr2
        