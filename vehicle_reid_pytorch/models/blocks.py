from torch import nn


def conv_block(num_in, num_out):
    return nn.Sequential(
        nn.BatchNorm2d(num_in),
        nn.ReLU(True),
        nn.Conv2d(num_in, num_out / 2, 1),
        nn.BatchNorm2d(num_out / 2),
        nn.ReLU(True),
        nn.Conv2d(num_out / 2, num_out / 2, 3, 1, 1),
        nn.BatchNorm2d(num_out / 2),
        nn.ReLU(True),
        nn.Conv2d(num_out / 2, num_out, 1)
    )


def skip_layer(num_in, num_out):
    if num_in == num_out:
        return Identity()
    else:
        return nn.Sequential(
            nn.Conv2d(num_in, num_out, 1, 1)
        )


class Residual(nn.Module):
    def __init__(self, num_in, num_out):
        super(Residual, self).__init__()
        self.conv_block = conv_block(num_in, num_out)
        self.skip_layer = skip_layer(num_in, num_out)

    def forward(self, x):
        return self.conv_block(x) + self.skip_layer(x)


class Identity(nn.Module):
    def forward(self, x, **kwargs):
        return x
