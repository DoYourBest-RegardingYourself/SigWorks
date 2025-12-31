"""
SNN Version of DeepGCN (Vision GNN) using SpikingJelly
将DeepGCN从ANN转换为SNN，使用SpikingJelly库实现脉冲神经网络
"""

import sys
sys.path.append(".")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from vanilla_vig.gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import load_pretrained
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model

# SpikingJelly imports
from spikingjelly.activation_based import neuron, layer, functional, surrogate, base


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

def spiking_act_layer(act, tau=2.0, detach_reset=True, backend='torch'):
    """
    返回脉冲神经元层，替代原来的激活函数
    
    Args:
        act: 激活类型（对于SNN，我们统一使用LIF神经元）
        tau: 膜电位时间常数
        detach_reset: 是否在反向传播时分离重置操作
        backend: 计算后端
    """
    act = act.lower()
    # 使用Leaky Integrate-and-Fire (LIF) 神经元
    return neuron.LIFNode(
        tau=tau,
        detach_reset=detach_reset,
        step_mode='m',  # 多步模式
        backend=backend,
        surrogate_function=surrogate.ATan()  # 使用ATan替代梯度
    )


class SpikingFFN(base.MemoryModule):
    """
    Spiking Feed-Forward Network
    将FFN模块转换为脉冲版本
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, tau=2.0, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 使用SpikingJelly的layer模块来包装卷积层
        self.fc1 = nn.Sequential(
            layer.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, step_mode='m'),
            layer.BatchNorm2d(hidden_features, step_mode='m'),
        )
        # 脉冲神经元替代激活函数
        self.sn1 = neuron.LIFNode(
            tau=tau,
            detach_reset=True,
            step_mode='m',
            surrogate_function=surrogate.ATan()
        )
        
        self.fc2 = nn.Sequential(
            layer.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, step_mode='m'),
            layer.BatchNorm2d(out_features, step_mode='m'),
        )
        # 第二个脉冲神经元
        self.sn2 = neuron.LIFNode(
            tau=tau,
            detach_reset=True,
            step_mode='m',
            surrogate_function=surrogate.ATan()
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x shape: [T, B, C, H, W] 其中T是时间步
        shortcut = x
        x = self.fc1(x)
        x = self.sn1(x)
        x = self.fc2(x)
        x = self.sn2(x)
        x = self.drop_path(x) + shortcut
        return x


class SpikingStem(base.MemoryModule):
    """
    Spiking Image to Visual Word Embedding
    将Stem模块转换为脉冲版本
    """
    def __init__(self, img_size=224, in_dim=1, out_dim=768, tau=2.0):
        super().__init__()
        
        self.conv1 = layer.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1, step_mode='m')
        self.bn1 = layer.BatchNorm2d(out_dim//8, step_mode='m')
        self.sn1 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m', 
                                   surrogate_function=surrogate.ATan())
        
        self.conv2 = layer.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1, step_mode='m')
        self.bn2 = layer.BatchNorm2d(out_dim//4, step_mode='m')
        self.sn2 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m',
                                   surrogate_function=surrogate.ATan())
        
        self.conv3 = layer.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1, step_mode='m')
        self.bn3 = layer.BatchNorm2d(out_dim//2, step_mode='m')
        self.sn3 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m',
                                   surrogate_function=surrogate.ATan())
        
        self.conv4 = layer.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1, step_mode='m')
        self.bn4 = layer.BatchNorm2d(out_dim, step_mode='m')
        self.sn4 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m',
                                   surrogate_function=surrogate.ATan())
        
        self.conv5 = layer.Conv2d(out_dim, out_dim, 3, stride=1, padding=1, step_mode='m')
        self.bn5 = layer.BatchNorm2d(out_dim, step_mode='m')
        # 最后一层不加脉冲神经元，保持特征

    def forward(self, x):
        # x shape: [T, B, C, H, W]
        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.sn3(self.bn3(self.conv3(x)))
        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        return x


class SpikingGrapher(base.MemoryModule):
    """
    Spiking Grapher module
    将Grapher模块转换为脉冲版本
    注意: 由于图卷积操作的特殊性，我们在Grapher外围添加脉冲层
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, 
                 relative_pos=False, tau=2.0):
        super().__init__()
        
        self.channels = in_channels
        self.n = n
        self.r = r
        
        # 输入脉冲神经元
        self.sn_in = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m',
                                     surrogate_function=surrogate.ATan())
        
        # FC1 层
        self.fc1 = nn.Sequential(
            layer.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
        )
        self.sn1 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m',
                                   surrogate_function=surrogate.ATan())
        
        # 图卷积层 - 这里我们需要一个特殊的处理
        # 因为Grapher内部有复杂的图操作，我们创建一个包装器
        self.graph_conv = SpikingDyGraphConv2d(
            in_channels, in_channels * 2, kernel_size, dilation, conv,
            norm, bias, stochastic, epsilon, r, tau=tau
        )
        
        # FC2 层
        self.fc2 = nn.Sequential(
            layer.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
        )
        self.sn2 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m',
                                   surrogate_function=surrogate.ATan())
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x shape: [T, B, C, H, W]
        _tmp = x
        x = self.sn_in(x)
        x = self.fc1(x)
        x = self.sn1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.sn2(x)
        x = self.drop_path(x) + _tmp
        return x


class SpikingDyGraphConv2d(base.MemoryModule):
    """
    Spiking Dynamic Graph Convolution
    将动态图卷积转换为脉冲版本
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, tau=2.0):
        super().__init__()
        
        from vanilla_vig.gcn_lib.torch_edge import DenseDilatedKnnGraph
        from vanilla_vig.gcn_lib.torch_nn import BasicConv, batched_index_select
        
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        
        # 使用脉冲版本的图卷积
        if conv == 'edge':
            self.gconv = SpikingEdgeConv2d(in_channels, out_channels, tau=tau, norm=norm, bias=bias)
        elif conv == 'mr':
            self.gconv = SpikingMRConv2d(in_channels, out_channels, tau=tau, norm=norm, bias=bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, relative_pos=None):
        # x shape: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        
        # 在时间维度上处理
        outputs = []
        for t in range(T):
            x_t = x[t]  # [B, C, H, W]
            y = None
            if self.r > 1:
                y = F.avg_pool2d(x_t, self.r, self.r)
                y = y.reshape(B, C, -1, 1).contiguous()
            x_t_reshaped = x_t.reshape(B, C, -1, 1).contiguous()
            edge_index = self.dilated_knn_graph(x_t_reshaped, y, relative_pos)
            out_t = self.gconv(x_t_reshaped, edge_index, y)
            outputs.append(out_t.reshape(B, -1, H, W))
        
        return torch.stack(outputs, dim=0)  # [T, B, C, H, W]


class SpikingMRConv2d(base.MemoryModule):
    """
    Spiking Max-Relative Graph Convolution
    """
    def __init__(self, in_channels, out_channels, tau=2.0, norm=None, bias=True):
        super().__init__()
        from vanilla_vig.gcn_lib.torch_nn import batched_index_select
        self.batched_index_select = batched_index_select
        
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=bias, groups=4),
            nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.Identity(),
        )
        self.sn = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='s',
                                  surrogate_function=surrogate.ATan())

    def forward(self, x, edge_index, y=None):
        x_i = self.batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = self.batched_index_select(y, edge_index[0])
        else:
            x_j = self.batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        x = self.nn(x)
        x = self.sn(x)
        return x


class SpikingEdgeConv2d(base.MemoryModule):
    """
    Spiking Edge Convolution
    """
    def __init__(self, in_channels, out_channels, tau=2.0, norm=None, bias=True):
        super().__init__()
        from vanilla_vig.gcn_lib.torch_nn import batched_index_select
        self.batched_index_select = batched_index_select
        
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=bias, groups=4),
            nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.Identity(),
        )
        self.sn = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='s',
                                  surrogate_function=surrogate.ATan())

    def forward(self, x, edge_index, y=None):
        x_i = self.batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = self.batched_index_select(y, edge_index[0])
        else:
            x_j = self.batched_index_select(x, edge_index[0])
        x = torch.cat([x_i, x_j - x_i], dim=1)
        x = self.nn(x)
        x = self.sn(x)
        max_value, _ = torch.max(x, -1, keepdim=True)
        return max_value


class SpikingDeepGCN(base.MemoryModule):
    """
    Spiking DeepGCN - SNN版本的视觉图神经网络
    
    Args:
        opt: 配置选项
        T: 时间步数 (默认4)
        tau: LIF神经元的膜电位时间常数 (默认2.0)
    """
    def __init__(self, opt, T=4, tau=2.0):
        super(SpikingDeepGCN, self).__init__()
        
        self.T = T
        self.tau = tau
        
        channels = opt.n_filters
        k = opt.k
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        
        # 脉冲版本的Stem
        self.stem = SpikingStem(out_dim=channels, tau=tau)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        
        # 脉冲版本的Backbone
        if opt.use_dilation:
            self.backbone = nn.ModuleList([
                nn.Sequential(
                    SpikingGrapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, norm,
                                   bias, stochastic, epsilon, 1, drop_path=dpr[i], tau=tau),
                    SpikingFFN(channels, channels * 4, drop_path=dpr[i], tau=tau)
                ) for i in range(self.n_blocks)
            ])
        else:
            self.backbone = nn.ModuleList([
                nn.Sequential(
                    SpikingGrapher(channels, num_knn[i], 1, conv, norm,
                                   bias, stochastic, epsilon, 1, drop_path=dpr[i], tau=tau),
                    SpikingFFN(channels, channels * 4, drop_path=dpr[i], tau=tau)
                ) for i in range(self.n_blocks)
            ])
        
        # 特征提取头 (输出1280维特征，与训练脚本中的分类层匹配)
        self.feature_head = nn.Sequential(
            layer.Conv2d(channels, 1280, 1, bias=True, step_mode='m'),
            layer.BatchNorm2d(1280, step_mode='m'),
            neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m',
                          surrogate_function=surrogate.ATan()),
        )
        
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, layer.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    @torch.compile
    def forward(self, inputs):
        """
        Args:
            inputs: [B, C, H, W] 输入图像
        
        Returns:
            输出特征向量 [B, 1280]
        """
        # 将输入扩展为时间序列 [T, B, C, H, W]
        x = inputs.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        # 通过Stem
        x = self.stem(x)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        T, B, C, H, W = x.shape
        
        # 通过Backbone
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
        
        # 全局平均池化
        x = x.mean(dim=[3, 4], keepdim=True)  # [T, B, C, 1, 1]
        
        # 特征提取
        x = self.feature_head(x)  # [T, B, 1280, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [T, B, 1280]
        
        # 对时间维度取平均（脉冲发放率编码）
        x = x.mean(dim=0)  # [B, 1280]
        
        return x

    @torch.compiler.disable(recursive=True)
    def reset(self):
        """重置所有脉冲神经元的状态"""
        for m in self.modules():
            if hasattr(m, 'reset') and m is not self:
                if isinstance(m, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                    m.reset()


@register_model
def spiking_vig_ti_224(pretrained=False, T=4, tau=2.0, **kwargs):
    """
    Spiking ViG Tiny
    """
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'  # 仅用于兼容，实际使用LIF神经元
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 12
            self.n_filters = 192
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = SpikingDeepGCN(opt, T=T, tau=tau)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def spiking_vig_s_224(pretrained=False, T=4, tau=2.0, **kwargs):
    """
    Spiking ViG Small
    """
    class OptInit:
        def __init__(self, num_classes=1280, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 16
            self.n_filters = 320
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = SpikingDeepGCN(opt, T=T, tau=tau)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def spiking_vig_b_224(pretrained=False, T=4, tau=2.0, **kwargs):
    """
    Spiking ViG Base
    """
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 16
            self.n_filters = 640
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = SpikingDeepGCN(opt, T=T, tau=tau)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


if __name__ == '__main__':
    # 测试模型
    model = spiking_vig_ti_224(num_classes=10, T=4, tau=2.0)
    print(model)
    
    # 测试前向传播
    x = torch.randn(2, 1, 224, 224)
    model.reset()  # 重置神经元状态
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
