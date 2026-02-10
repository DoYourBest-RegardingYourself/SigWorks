import torch
from torch import nn
from collections import OrderedDict
from spikingjelly.activation_based import neuron, layer, functional, surrogate


# ===================== SNN 辅助函数 =====================

def conv_bn_lif(in_channels, out_channels, kernel_size, stride=1, pad=0, tau=2.0):
    """
    卷积 + BatchNorm + LIF神经元 模块 (用于SNN)
    使用SpikingJelly的LIF神经元替代ReLU
    """
    return nn.Sequential(OrderedDict([
        ('conv', layer.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', layer.BatchNorm2d(out_channels)),
        ('lif', neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())),
    ]))


def linear_bn_lif(in_features, out_features, tau=2.0):
    """
    全连接 + BatchNorm + LIF神经元 模块 (用于SNN)
    """
    return nn.Sequential(OrderedDict([
        ('fc', layer.Linear(in_features, out_features, bias=False)),
        ('bn', layer.BatchNorm1d(out_features)),
        ('lif', neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())),
    ]))


# ===================== 原始ANN辅助函数 (保留用于对比) =====================

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    """原始的卷积+BN+ReLU模块"""
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def linear_bn_relu(in_features, out_features):
    """原始的全连接+BN+ReLU模块"""
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))


# ===================== SNN 模型 =====================

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


class SigNetSNN(nn.Module):
    """
    SigNet SNN模型 - 使用SpikingJelly框架
    基于 https://arxiv.org/abs/1705.05787 的SNN版本
    """

    def __init__(self, num_classes=10, T=4, tau=2.0):
        """
        Args:
            num_classes: 分类类别数
            T: 时间步数 (SNN仿真步数)
            tau: LIF神经元的时间常数
        """
        super(SigNetSNN, self).__init__()

        self.T = T
        self.feature_space_size = 2048

        # 编码层：将静态图像编码为脉冲序列
        self.encoder = layer.Conv2d(1, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.encoder.weight)
        
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_lif(1, 96, 11, stride=4, tau=tau)),
            ('maxpool1', layer.MaxPool2d(3, 2)),
            ('conv2', conv_bn_lif(96, 256, 5, pad=2, tau=tau)),
            ('maxpool2', layer.MaxPool2d(3, 2)),
            ('conv3', conv_bn_lif(256, 384, 3, pad=1, tau=tau)),
            ('conv4', conv_bn_lif(384, 384, 3, pad=1, tau=tau)),
            ('conv5', conv_bn_lif(384, 256, 3, pad=1, tau=tau)),
            ('maxpool3', layer.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_lif(6400, 2048, tau=tau)),
            ('fc2', linear_bn_lif(self.feature_space_size, self.feature_space_size, tau=tau)),
        ]))
        
        self.fc = layer.Linear(self.feature_space_size, num_classes)

    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs: 输入图像 [B, C, H, W]
        Returns:
            脉冲发放率 [B, feature_space_size]
        """
        # 重置所有神经元状态
        functional.reset_net(self)
        
        # 累积多个时间步的输出
        spike_out = 0
        
        for t in range(self.T):
            x = self.encoder(inputs)
            x = self.conv_layers(x)
            flattened_size = x.view(x.size(0), -1).shape[1]
            x = x.view(x.size(0), flattened_size)
            x = self.fc_layers(x)
            spike_out = spike_out + x
        
        # 返回平均脉冲发放率
        return spike_out / self.T


class SigNetSNN_thin(nn.Module):
    """SigNet_thin 的 SNN 版本"""
    
    def __init__(self, T=4, tau=2.0):
        super(SigNetSNN_thin, self).__init__()

        self.T = T
        self.feature_space_size = 1024

        self.encoder = layer.Conv2d(1, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.encoder.weight)

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_lif(1, 96, 11, stride=4, tau=tau)),
            ('maxpool1', layer.MaxPool2d(3, 2)),
            ('conv2', conv_bn_lif(96, 128, 5, pad=2, tau=tau)),
            ('maxpool2', layer.MaxPool2d(3, 2)),
            ('conv3', conv_bn_lif(128, 128, 3, pad=1, tau=tau)),
            ('conv4', conv_bn_lif(128, 128, 3, pad=1, tau=tau)),
            ('conv5', conv_bn_lif(128, 128, 3, pad=1, tau=tau)),
            ('maxpool3', layer.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_lif(128 * 3 * 5, self.feature_space_size, tau=tau)),
            ('fc2', linear_bn_lif(self.feature_space_size, self.feature_space_size, tau=tau)),
        ]))

    def forward(self, inputs):
        functional.reset_net(self)
        
        spike_out = 0
        for t in range(self.T):
            x = self.encoder(inputs)
            x = self.conv_layers(x)
            x = x.view(x.shape[0], 128 * 3 * 5)
            x = self.fc_layers(x)
            spike_out = spike_out + x
        
        return spike_out / self.T


class SigNetSNN_smaller(nn.Module):
    """SigNet_smaller 的 SNN 版本"""
    
    def __init__(self, T=4, tau=2.0):
        super(SigNetSNN_smaller, self).__init__()

        self.T = T
        self.feature_space_size = 2048

        self.encoder = layer.Conv2d(1, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.encoder.weight)

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_lif(1, 96, 11, stride=4, tau=tau)),
            ('maxpool1', layer.MaxPool2d(3, 2)),
            ('conv2', conv_bn_lif(96, 256, 5, pad=2, tau=tau)),
            ('maxpool2', layer.MaxPool2d(3, 2)),
            ('conv3', conv_bn_lif(256, 384, 3, pad=1, tau=tau)),
            ('conv5', conv_bn_lif(384, 256, 3, pad=1, tau=tau)),
            ('maxpool3', layer.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_lif(256 * 3 * 5, 2048, tau=tau)),
        ]))

    def forward(self, inputs):
        functional.reset_net(self)
        
        spike_out = 0
        for t in range(self.T):
            x = self.encoder(inputs)
            x = self.conv_layers(x)
            x = x.view(x.shape[0], 256 * 3 * 5)
            x = self.fc_layers(x)
            spike_out = spike_out + x
        
        return spike_out / self.T


# ===================== 小波增强 SNN 模型 =====================

class SigNetSNN_Wavelet(nn.Module):
    """
    支持小波频率增强的SigNet SNN模型
    
    接收格式: [B, T, C, H, W] - 每个时间步包含原图+对应频率成分
    其中 C = 2 (原始灰度图 + 频率成分)
    """
    
    def __init__(self, num_classes=10, T=4, tau=2.0, in_channels=2):
        """
        Args:
            num_classes: 分类类别数
            T: 时间步数 (SNN仿真步数)
            tau: LIF神经元的时间常数
            in_channels: 输入通道数 (原图+频率成分)
        """
        super(SigNetSNN_Wavelet, self).__init__()
        
        self.T = T
        self.in_channels = in_channels
        self.feature_space_size = 2048
        
        # 输入融合层：将多通道输入融合
        self.input_fusion = nn.Sequential(
            layer.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(32),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Conv2d(32, 1, kernel_size=1, bias=False),
        )
        
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_lif(1, 96, 11, stride=4, tau=tau)),
            ('maxpool1', layer.MaxPool2d(3, 2)),
            ('conv2', conv_bn_lif(96, 256, 5, pad=2, tau=tau)),
            ('maxpool2', layer.MaxPool2d(3, 2)),
            ('conv3', conv_bn_lif(256, 384, 3, pad=1, tau=tau)),
            ('conv4', conv_bn_lif(384, 384, 3, pad=1, tau=tau)),
            ('conv5', conv_bn_lif(384, 256, 3, pad=1, tau=tau)),
            ('maxpool3', layer.MaxPool2d(3, 2)),
        ]))
        
        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_lif(6400, 2048, tau=tau)),
            ('fc2', linear_bn_lif(self.feature_space_size, self.feature_space_size, tau=tau)),
        ]))
        
        self.fc = layer.Linear(self.feature_space_size, num_classes)
    
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 
                - 如果启用小波增强: [B, T, C, H, W] 每个时间步有不同的频率输入
                - 如果未启用: [B, C, H, W] 所有时间步使用相同输入
        
        Returns:
            脉冲发放率 [B, feature_space_size]
        """
        functional.reset_net(self)
        
        # 检查输入格式
        if len(inputs.shape) == 5:
            # [B, T, C, H, W] - 小波增强格式
            B, T, C, H, W = inputs.shape
            assert T == self.T, f"Input timesteps {T} != model timesteps {self.T}"
            wavelet_mode = True
        else:
            # [B, C, H, W] - 标准格式
            wavelet_mode = False
        
        spike_out = 0
        
        for t in range(self.T):
            if wavelet_mode:
                # 使用该时间步对应的输入（包含对应频率成分）
                x_t = inputs[:, t, :, :, :]  # [B, C, H, W]
            else:
                x_t = inputs
            
            # 融合多通道输入
            x = self.input_fusion(x_t)
            
            # 主干网络
            x = self.conv_layers(x)
            flattened_size = x.view(x.size(0), -1).shape[1]
            x = x.view(x.size(0), flattened_size)
            x = self.fc_layers(x)
            
            spike_out = spike_out + x
        
        return spike_out / self.T


class SigNetSNN_thin_Wavelet(nn.Module):
    """SigNet_thin 的小波增强 SNN 版本"""
    
    def __init__(self, T=4, tau=2.0, in_channels=2):
        super(SigNetSNN_thin_Wavelet, self).__init__()
        
        self.T = T
        self.in_channels = in_channels
        self.feature_space_size = 1024
        
        # 输入融合层
        self.input_fusion = nn.Sequential(
            layer.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(16),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Conv2d(16, 1, kernel_size=1, bias=False),
        )
        
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_lif(1, 96, 11, stride=4, tau=tau)),
            ('maxpool1', layer.MaxPool2d(3, 2)),
            ('conv2', conv_bn_lif(96, 128, 5, pad=2, tau=tau)),
            ('maxpool2', layer.MaxPool2d(3, 2)),
            ('conv3', conv_bn_lif(128, 128, 3, pad=1, tau=tau)),
            ('conv4', conv_bn_lif(128, 128, 3, pad=1, tau=tau)),
            ('conv5', conv_bn_lif(128, 128, 3, pad=1, tau=tau)),
            ('maxpool3', layer.MaxPool2d(3, 2)),
        ]))
        
        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_lif(128 * 3 * 5, self.feature_space_size, tau=tau)),
            ('fc2', linear_bn_lif(self.feature_space_size, self.feature_space_size, tau=tau)),
        ]))
    
    def forward(self, inputs):
        functional.reset_net(self)
        
        if len(inputs.shape) == 5:
            B, T, C, H, W = inputs.shape
            wavelet_mode = True
        else:
            wavelet_mode = False
        
        spike_out = 0
        
        for t in range(self.T):
            if wavelet_mode:
                x_t = inputs[:, t, :, :, :]
            else:
                x_t = inputs
            
            x = self.input_fusion(x_t)
            x = self.conv_layers(x)
            x = x.view(x.shape[0], 128 * 3 * 5)
            x = self.fc_layers(x)
            spike_out = spike_out + x
        
        return spike_out / self.T


# ===================== 原始 ANN 模型 (保留用于对比) =====================

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


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 测试 SNN 模型
    model = SigNetSNN(num_classes=10, T=4, tau=2.0)
    x = torch.randn(2, 1, 150, 220)  # 批量大小=2, 通道=1, 高=150, 宽=220
    output = model(x)
    print(f"SigNetSNN output shape: {output.shape}")
    
    # 测试 SNN thin 模型
    model_thin = SigNetSNN_thin(T=4, tau=2.0)
    output_thin = model_thin(x)
    print(f"SigNetSNN_thin output shape: {output_thin.shape}")
