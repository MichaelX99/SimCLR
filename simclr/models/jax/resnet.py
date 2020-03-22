"""
Lots taken from https://github.com/google/jax/blob/master/examples/resnet50.py
Jax is new to me so who knows whats going on here
"""

from jax.experimental import stax

class ResNet(object):
    def __init__(self, num_classes=100, encoding=True):

        blocks = [
            stax.GeneralConv(('HWCN', 'OIHW', 'NHWC'), 64, (7, 7), (2, 2), 'SAME'),
            stax.BatchNorm(), stax.Relu, stax.MaxPool((3, 3), strides=(2, 2)),
            self.ConvBlock(3, [64, 64, 256], strides=(1, 1)),
            self.IdentityBlock(3, [64, 64]),
            self.IdentityBlock(3, [64, 64]),
            self.ConvBlock(3, [128, 128, 512]),
            self.IdentityBlock(3, [128, 128]),
            self.IdentityBlock(3, [128, 128]),
            self.IdentityBlock(3, [128, 128]),
            self.ConvBlock(3, [256, 256, 1024]),
            self.IdentityBlock(3, [256, 256]),
            self.IdentityBlock(3, [256, 256]),
            self.IdentityBlock(3, [256, 256]),
            self.IdentityBlock(3, [256, 256]),
            self.IdentityBlock(3, [256, 256]),
            self.ConvBlock(3, [512, 512, 2048]),
            self.IdentityBlock(3, [512, 512]),
            self.IdentityBlock(3, [512, 512]),
            stax.AvgPool((7, 7))
        ]

        if not encoding:
            blocks.append(stax.Flatten)
            blocks.append(stax.Dense(num_classes))

        self.model = stax.serial(*blocks)

    def ConvBlock(self, kernel_size, filters, strides=(2,2)):
        filters1, filters2, filters3 = filters
        Main = stax.serial(
            stax.Conv(filters1, (1, 1), strides), stax.BatchNorm(), stax.Relu,
            stax.Conv(filters2, (kernel_size, kernel_size), padding='SAME'), stax.BatchNorm(), stax.Relu,
            stax.Conv(filters3, (1, 1)), stax.BatchNorm())
        Shortcut = stax.serial(stax.Conv(filters3, (1, 1), strides), stax.BatchNorm())
        return stax.serial(stax.FanOut(2), stax.parallel(Main, Shortcut), stax.FanInSum, stax.Relu)

    def IdentityBlock(self, kernel_size, filters):
        filters1, filters2 = filters
        def make_main(input_shape):
            # the number of output channels depends on the number of input channels
            return stax.serial(
                stax.Conv(filters1, (1, 1)), stax.BatchNorm(), stax.Relu,
                stax.Conv(filters2, (kernel_size, kernel_size), padding='SAME'), stax.BatchNorm(), stax.Relu,
                stax.Conv(input_shape[3], (1, 1)), stax.BatchNorm())
        Main = stax.shape_dependent(make_main)
        return stax.serial(stax.FanOut(2), stax.parallel(Main, stax.Identity), stax.FanInSum, stax.Relu)

    def forward(self, x):
        pass