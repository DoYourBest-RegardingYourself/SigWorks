from torch import nn
from collections import OrderedDict


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head
        self.true_fc = nn.Linear(2048, 2)  # 辨别真假的全连接层

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


class SigNet(nn.Module):
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """

    def __init__(self, num_classes=10):
        super(SigNet, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(256, 384, 3, pad=1)),
            ('conv4', conv_bn_relu(384, 384, 3, pad=1)),
            ('conv5', conv_bn_relu(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            # ('fc1', linear_bn_relu(256 * 3 * 5, 2048)),
            ('fc1', linear_bn_relu(6400, 2048)),
            ('fc2', linear_bn_relu(self.feature_space_size, self.feature_space_size)),
        ]))
        self.fc = nn.Linear(self.feature_space_size, num_classes)

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        flattened_size = x.view(x.size(0), -1).shape[1]
        x = x.view(x.size(0), flattened_size)
        x = self.fc_layers(x)
        return x

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        # ('conv', RepConv(in_channels, out_channels, kernel_size, stride, pad)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))


class SigNet_thin(nn.Module):
    def __init__(self):
        super(SigNet_thin, self).__init__()

        self.feature_space_size = 1024

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(96, 128, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(128, 128, 3, pad=1)),
            ('conv4', conv_bn_relu(128, 128, 3, pad=1)),
            ('conv5', conv_bn_relu(128, 128, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_relu(128 * 3 * 5, self.feature_space_size)),
            ('fc2', linear_bn_relu(self.feature_space_size, self.feature_space_size)),
        ]))

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = x.view(x.shape[0], 128 * 3 * 5)
        x = self.fc_layers(x)
        return x


class SigNet_smaller(nn.Module):
    def __init__(self):
        super(SigNet_smaller, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(256, 384, 3, pad=1)),
            ('conv5', conv_bn_relu(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_relu(256 * 3 * 5, 2048)),
        ]))

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.fc_layers(x)
        return x
