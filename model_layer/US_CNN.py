from torch import nn
from torch.nn import Conv2d as Conv2D
import math


class CNN1D(nn.Module):
    def __init__(
        self,
        in_channels,
        channels_out = 60,
        n_classes = 1,
        big_filter_kernel = (49,1),
        kernel_size = 5,
        depth = 3,
        dilation = 1,
        batch_norm = False,
    ):

        super().__init__()
        # assert up_mode in ('upconv', 'upsample')
        self.dilation = dilation
        self.depth = depth
        prev_channels = channels_out
        self.channels_out = channels_out
        self.big_filter_kernel = big_filter_kernel
        self.kernel_size = kernel_size
        self.increasing_depth = False
        self.batch_norm = batch_norm

        self.down_path = nn.ModuleList()
        self.down_path.append(
            Conv2D(in_channels, channels_out, kernel_size=self.big_filter_kernel,
                       padding=(math.floor(self.big_filter_kernel[0]/2), 0)))
        self.down_path.append(nn.ReLU(inplace=True))

        if depth>0:
            for i in range(depth):
                if self.increasing_depth:
                    next_channels = 2 ** (4 + i)
                else:
                    next_channels = prev_channels

                self.down_path.append(
                    Conv2D(prev_channels,next_channels,kernel_size=(kernel_size,1),padding=(math.floor(kernel_size/2)*(dilation),0),dilation=(dilation,1))
                )
                self.down_path.append(nn.ReLU(inplace=True))
                if batch_norm:
                    self.down_path.append(nn.BatchNorm2d(next_channels))

        self.down_path.append(nn.Dropout(0.2))
        self.last = nn.Conv2d(prev_channels,n_classes,kernel_size=(1,1) )

        for m in self.down_path.modules():
            if isinstance(m, Conv2D):
                nn.init.xavier_normal(m.weight)

    def forward(self, x):

        for i, down in enumerate(self.down_path):
            x = down(x)

        return self.last(x)