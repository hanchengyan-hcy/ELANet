from torch import nn


class SubSpace(nn.Module):
    """
    Subspace class.
    ...
    Attributes
    ----------
    nin : int
        number of input feature volume.
    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.
    """

    def __init__(self, nin):
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out

