from torch import nn

def double_conv(channels_in, channels_out):

    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=channels_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=channels_out),
        nn.ReLU(inplace=True),
    )


class SegNetS(nn.Module):
    def __init__(self, params):
        super().__init__()

        C_in = params['channels']
        out_channels = params['out_channels']

        self.enc_conv0 = double_conv(C_in,32)

        self.pool0 = nn.MaxPool2d(2)  # 256 -> 128

        self.enc_conv1 = double_conv(32,64)
        self.pool1 =   nn.MaxPool2d(2) # 128 -> 64
        self.enc_conv2 = double_conv(64,128)
        self.pool2 =   nn.MaxPool2d(2) # 64 -> 32
        self.enc_conv3 =  double_conv(128,256)
        self.pool3 =  nn.MaxPool2d(2) # 32 -> 16

        # bottleneck
        self.bottleneck_conv = double_conv(256,256)

        # decoder (upsampling)
        self.upsample0 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 16 -> 32
        self.dec_conv0 = double_conv(256,128)
        self.upsample1 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   # 32 -> 64
        self.dec_conv1 = double_conv(128,64)
        self.upsample2 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64 -> 128
        self.dec_conv2 = double_conv(64,32)
        self.upsample3 =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128 -> 256
        self.dec_conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        x= self.pool0(e0)
        e1 = self.enc_conv1(x)
        x= self.pool1(e1)
        e2 = self.enc_conv2(x)
        x= self.pool2(e2)
        e3 = self.enc_conv3(x)
        x= self.pool3(e3)

        # bottleneck
        b = self.bottleneck_conv(x)

        # decoder
        d0 = self.upsample0(x)
        x = self.dec_conv0(d0)
        d1 = self.upsample1(x)
        x = self.dec_conv1(d1)
        d2 =  self.upsample2(x)
        x = self.dec_conv2(d2)
        d3 =  self.upsample3(x)
        
        out = self.dec_conv3(d3) # no activation
        return out