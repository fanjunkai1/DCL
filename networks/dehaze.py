import torch.nn as nn
import torch
import functools
import numpy as np
import torch.nn.functional as F



def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.

    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project
    (https://github.com/jcjohnson/fast-neural-style)

    """
    def __init__(self, 
                 input_nc = 3,
                 output_nc = 3,
                 ngf = 64, 
                 norm = 'instance', 
                 use_dropout=False, 
                 n_blocks=6, 
                 padding_type='reflect' 
                 ):
        """
        Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 
                           ngf, 
                           kernel_size=7, 
                           padding=0, 
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        n_downsampling = 2

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, 
                                ngf * mult * 2, 
                                kernel_size=3, 
                                stride=2, 
                                padding=1, 
                                bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, 
                                  padding_type=padding_type, 
                                  norm_layer=norm_layer, 
                                  use_dropout=use_dropout, 
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, 
            #                              int(ngf * mult / 2),
            #                              kernel_size=3, 
            #                              stride=2,
            #                              padding=1, 
            #                              output_padding=1,
            #                              bias=use_bias),
            #           norm_layer(int(ngf * mult / 2)),
            #           nn.ReLU(True)]
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, padding=1),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
                #Printer('upsample %d'%mult)
            ]
        # model += [
        # nn.PixelShuffle(upscale_factor=2),
        # # nn.Upsample(scale_factor=2, mode='nearest'),
        # nn.Conv2d(int(ngf / 2), int(ngf/4), 3, padding=1),
        # norm_layer(int(ngf/4)),
        # nn.ReLU(True),
        # #Printer('upsample %d'%mult)
        # ]
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc , kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        return output


class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    def __init__(self, 
                 dim, 
                 padding_type, 
                 norm_layer, 
                 use_dropout, 
                 use_bias):

        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, 
                                                padding_type, 
                                                norm_layer, 
                                                use_dropout, 
                                                use_bias)

    def build_conv_block(self, 
                         dim, 
                         padding_type, 
                         norm_layer, 
                         use_dropout, 
                         use_bias):

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, 
                                 dim, 
                                 kernel_size=3, 
                                 stride=1, 
                                 padding=p, 
                                 bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, 
                                 dim, 
                                 kernel_size=3, 
                                 stride=1, 
                                 padding=p, 
                                 bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out




def get_dark_channel(img, neighborhood_size = 15):

    shape = img.shape
    if len(shape) == 4:
        img_min,_ = torch.min(img, dim=1)

        padSize = np.int(np.floor(neighborhood_size/2))
        if neighborhood_size % 2 == 0:
            pads = [padSize, padSize-1 ,padSize ,padSize-1]
        else:
            pads = [padSize, padSize, padSize, padSize]

        img_min = F.pad(img_min, pads, mode='constant', value=1)
        dark_img = -F.max_pool2d(-img_min, kernel_size=neighborhood_size, stride=1)
    else:
        raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

    dark_img = torch.unsqueeze(dark_img, dim=1)
    return dark_img



def get_atmospheric_light(img, dark_img):

    num, chl, height, width = img.shape
    topNum = np.int(0.01 * height * width)

    A_infinte = torch.Tensor(num, chl, 1, 1)
    if img.is_cuda:
        A_infinte = A_infinte.cuda()

    for num_id in range(num):
        curImg = img[num_id,...]
        curDarkImg = dark_img[num_id,0,...]

        _, indices = curDarkImg.reshape([height*width]).sort(descending=True)
        #curMask = indices < topNum

        for chl_id in range(chl):
            imgSlice = curImg[chl_id,...].reshape([height*width])
            A_infinte[num_id,chl_id,0,0] = torch.mean(imgSlice[indices[0:topNum]])

    return A_infinte


class NLayerDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator

    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm='instance'):
        """
        Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer

        """
        super(NLayerDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type = norm)

        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:  
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, 
                              ndf, 
                              kernel_size=kw, 
                              stride=2, 
                              padding=padw
                              ), 
                    nn.LeakyReLU(0.2, True)
                    ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev,
                          ndf * nf_mult, 
                          kernel_size=kw, 
                          stride=2, 
                          padding=padw, 
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, 
                      ndf * nf_mult, 
                      kernel_size=kw, 
                      stride=1, 
                      padding=padw, 
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 
                                1, 
                                kernel_size=kw, 
                                stride=1, 
                                padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    return LH, HL, HH


def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    # harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    # filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    # LL = net(in_channels, in_channels,
    #          kernel_size=2, stride=2, padding=0, bias=False,
    #          groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    # LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    # LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat3'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LH, HL, HH):
        if self.option_unpool == 'sum':
            return self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat3':
            return torch.cat([self.LH(LH), self.HL(HL), self.HH(HH)], dim=1)
        else:
            raise NotImplementedError
        

class ResnetGeneratorFS(nn.Module):
    def __init__(self,  input_nc = 3, output_nc = 3, ngf = 64, norm = 'instance', use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGeneratorFS, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.Convlayers1 = nn.Sequential(*[
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)])

        self.downsample1 = nn.Sequential(*[
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)])

        self.downsample2 = nn.Sequential(*[
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)])

        self.ResnetBlock = []
        for i in range(n_blocks):  # add ResNet blocks
            self.ResnetBlock.append(
                ResnetBlock(ngf * 4, 
                            padding_type=padding_type, 
                            norm_layer=norm_layer, 
                            use_dropout=use_dropout,
                            use_bias=use_bias))
            
        self.ResnetBlock = nn.Sequential(*self.ResnetBlock)

        self.upsample1 = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * 12, int(ngf * 2), 3, padding=1),
                norm_layer(int(ngf * 12 / 2)),
                nn.ReLU(True),
                #Printer('upsample %d'%mult)
            ])

        self.upsample2 = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * 6, int(ngf), 3, padding=1),
                norm_layer(int(ngf * 6 / 2)),
                nn.ReLU(True),
                #Printer('upsample %d'%mult)
            ])

        self.Convlayers2 = nn.Sequential(*[
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf * 3, output_nc, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf * 3),
            nn.Tanh()])
        
        # WavePool
        self.pool64 = WavePool(64).cuda()
        self.pool128 = WavePool(128).cuda()
        self.pool256 = WavePool(256).cuda()

        # WaveUnpool
        self.recon_block1 = WaveUnpool(128, "sum")
        self.recon_block2 = WaveUnpool(256, "sum")
        self.recon_block3 = WaveUnpool(512, "sum")
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)
        


    def forward(self, x):

        output_l1 = self.Convlayers1(x)
        LH_1, HL_1, HH_1 = self.pool64(output_l1)
        original_1 = self.recon_block1(LH_1, HL_1, HH_1)
        output_d1 = self.downsample1(output_l1)

        LH_2, HL_2, HH_2 = self.pool128(output_d1)
        original_2 = self.recon_block2(LH_2, HL_2, HH_2)
        output_d2 = self.downsample2(output_d1)

        LH_3, HL_3, HH_3 = self.pool256(output_d2)
        original_3 = self.recon_block3(LH_3, HL_3, HH_3)

        output_b = self.ResnetBlock(output_d2)

        output_b = torch.cat((output_b, original_3), dim=1)
        output_u1 = self.upsample1(output_b)

        output_u1 = torch.cat((output_u1, original_2), dim=1)
        
        output_u2 = self.upsample2(output_u1)
        output_u2 = torch.cat((output_u2, original_1), dim=1)
        
        output = self.Convlayers2(output_u2)

        return output
    


class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    

class HFDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator

    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm='instance'):
        """
        Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer

        """
        super(HFDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type = norm)

        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:  
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc*7, 
                              ndf, 
                              kernel_size=kw, 
                              stride=2, 
                              padding=padw
                              ), 
                    nn.LeakyReLU(0.2, True)
                    ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev,
                          ndf * nf_mult, 
                          kernel_size=kw, 
                          stride=2, 
                          padding=padw, 
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, 
                      ndf * nf_mult, 
                      kernel_size=kw, 
                      stride=1, 
                      padding=padw, 
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 
                                1, 
                                kernel_size=kw, 
                                stride=1, 
                                padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.apply(self._init_weights)

        # WavePool
        self.pool64 = WavePool(input_nc)

        # WaveUnpool
        self.recon_block1 = WaveUnpool(input_nc*2, "cat3")
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False )


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, input):
        """Standard forward."""
        LH, HL, HH = self.pool64(input)
        frequency = torch.cat([LH, HL, HH], dim=1)
        frequency = self.upsamplex2(frequency)
        inputs = torch.cat([frequency, input], dim=1)
        
        return self.model(inputs)
