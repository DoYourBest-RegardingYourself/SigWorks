import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq

# ---- PyTorch Geometric imports ----
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import knn_graph

# ---- timm imports (kept for compatibility with original repo) ----
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model

# ---------------------------------------------------------------
# Basic config (same style as original vig.py)
# ---------------------------------------------------------------


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "gnn_patch16_224": _cfg(
        crop_pct=0.9,
        input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


# ---------------------------------------------------------------
# Activation layer helper (mimic original gcn_lib.act_layer)
# ---------------------------------------------------------------


def act_layer(act: str = "relu", inplace: bool = True):
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace=inplace)
    elif act == "prelu":
        return nn.PReLU()
    elif act == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
    elif act == "gelu":
        return nn.GELU()
    elif act == "hswish":
        return nn.Hardswish(inplace=inplace)
    else:
        raise ValueError(f"Unsupported act type: {act}")


# ---------------------------------------------------------------
# FFN block (same as original, conv-FFN in 2D)
# ---------------------------------------------------------------


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act="relu",
        drop_path=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        x = x + shortcut
        return x


# ---------------------------------------------------------------
# Stem (image -> visual tokens), same as original idea
# ---------------------------------------------------------------


class Stem(nn.Module):
    """
    Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 8),
            act_layer(act),
            nn.Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 4),
            act_layer(act),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


# ---------------------------------------------------------------
# PyG-based Grapher block
# ---------------------------------------------------------------


class NeoVisionGNN(nn.Module):
    """
    A PyTorch Geometric-based graph convolution block that conceptually
    replaces the original Grapher module in Vision GNN.

    Pipeline:
    - Input feature map: B x C x H x W
    - Flatten to graph: (B*H*W) nodes, each with C-dim features
    - Build kNN graph in feature space using PyG knn_graph
    - Apply GCNConv
    - Reshape back to B x C x H x W
    - Residual + BN + activation + DropPath (similar style to original)
    """

    def __init__(
        self,
        channels: int,
        k: int,
        act: str = "gelu",
        norm: str = "batch",
        bias: bool = True,
        epsilon: float = 0.0,
        stochastic: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.k = k
        self.epsilon = epsilon
        self.stochastic = stochastic  # kept for interface consistency

        # GCNConv as a simple message passing layer
        self.gcn = GCNConv(
            in_channels=channels,
            out_channels=channels,
            bias=bias,
            add_self_loops=True,
            normalize=True,
        )

        # Normalization on 2D feature map
        if norm == "batch":
            self.norm = nn.BatchNorm2d(channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        else:
            raise ValueError(f"Unsupported norm type: {norm}")

        self.act = act_layer(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @staticmethod
    def _build_batch_index(batch_size: int, num_nodes_per_graph: int, device):
        """
        Build PyG batch index tensor of shape [B * N], where each node knows
        which example in the batch it belongs to.
        """
        batch = torch.arange(batch_size, device=device).view(-1, 1)
        batch = batch.repeat(1, num_nodes_per_graph).view(-1)
        return batch

    def forward(self, x: torch.Tensor):
        """
        x: B x C x H x W
        """
        B, C, H, W = x.shape
        shortcut = x

        # B x C x H x W -> B x H x W x C -> (B*H*W) x C
        x_nodes = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)

        # Batch index for PyG
        batch = self._build_batch_index(B, H * W, x.device)

        # kNN graph in feature space
        # Note: we use feature x_nodes to build dynamic graph per forward.
        edge_index = knn_graph(x_nodes, k=self.k, batch=batch, loop=False)

        # GCNConv requires (N, F) features
        x_out = self.gcn(x_nodes, edge_index)  # (B*H*W) x C

        # Optional stochastic residual scaling (epsilon)
        if self.stochastic and self.epsilon > 0.0 and self.training:
            # Simple variant: randomly drop graph update
            mask = torch.rand(1, device=x.device) >= self.epsilon
            x_out = mask * x_out + (~mask) * x_nodes

        # Reshape back: (B*H*W) x C -> B x H x W x C -> B x C x H x W
        x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Norm + activation + DropPath + residual
        x_out = self.norm(x_out)
        x_out = self.act(x_out)
        x_out = self.drop_path(x_out) + shortcut

        return x_out


# ---------------------------------------------------------------
# DeepGCN backbone with PyG-based Grapher
# ---------------------------------------------------------------


class DeepGCN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path

        # Stem
        self.stem = Stem(out_dim=channels, act=act)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        print("dpr", dpr)

        # Number of kNN neighbors per block (linear from k to 2k)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)]
        print("num_knn", num_knn)

        # Position embedding kept as in original (14x14 for 224x224)
        # If input size changes, this may need adjustment.
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        # Backbone: [PyGGrapher -> FFN] * n_blocks
        blocks = []
        for i in range(self.n_blocks):
            graph_block = NeoVisionGNN(
                channels=channels,
                k=num_knn[i],
                act=act,
                norm=norm,
                bias=bias,
                epsilon=epsilon,
                stochastic=stochastic,
                drop_path=dpr[i],
            )
            ffn_block = FFN(
                in_features=channels,
                hidden_features=channels * 4,
                act=act,
                drop_path=dpr[i],
            )
            blocks.append(Seq(graph_block, ffn_block))
        self.backbone = Seq(*blocks)

        # Classification head
        self.prediction = Seq(
            nn.Conv2d(channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(opt.dropout),
            nn.Conv2d(1024, opt.n_classes, 1, bias=True),
        )

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs: torch.Tensor):
        """
        inputs: B x 3 x H x W (H=W=224 for default cfg)
        """
        x = self.stem(inputs)
        # Add learnable position embedding, broadcast over batch
        x = x + self.pos_embed

        B, C, H, W = x.shape  # H=W=14 if input is 224x224

        # Sequential Grapher+FFN blocks
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        # Global pooling + classifier
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.prediction(x).squeeze(-1).squeeze(-1)  # B x num_classes
        return x


# ---------------------------------------------------------------
# Model factory functions (same names as original vig.py)
# ---------------------------------------------------------------


@register_model
def vig_ti_224_gelu(pretrained: bool = False, **kwargs):
    class OptInit:
        def __init__(
            self,
            num_classes=1000,
            drop_path_rate=0.0,
            drop_rate=0.0,
            num_knn=9,
            **_kwargs,
        ):
            # graph hyper-parameters
            self.k = num_knn  # neighbor num (default: 9)
            self.conv = "mr"  # kept for compatibility, unused here
            self.act = "gelu"  # activation layer
            self.norm = "batch"  # {batch, instance}
            self.bias = True  # bias of conv layer
            self.n_blocks = 12  # number of basic blocks
            self.n_filters = 192  # channels of deep features
            self.n_classes = num_classes  # output dimension
            self.dropout = drop_rate  # dropout rate
            self.use_dilation = True  # unused here, kept for compat
            self.epsilon = 0.2  # stochastic epsilon
            self.use_stochastic = False  # whether to use stochastic residual
            self.drop_path = drop_path_rate  # global drop path rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["gnn_patch16_224"]
    # Not loading pretrained weights here; you can adapt a conversion script if needed.
    return model


@register_model
def vig_s_224_gelu(pretrained: bool = False, **kwargs):
    class OptInit:
        def __init__(
            self,
            num_classes=1000,
            drop_path_rate=0.0,
            drop_rate=0.0,
            num_knn=9,
            **_kwargs,
        ):
            self.k = num_knn
            self.conv = "mr"
            self.act = "gelu"
            self.norm = "batch"
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
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["gnn_patch16_224"]
    return model


@register_model
def vig_b_224_gelu(pretrained: bool = False, **kwargs):
    class OptInit:
        def __init__(
            self,
            num_classes=1000,
            drop_path_rate=0.0,
            drop_rate=0.0,
            num_knn=9,
            **_kwargs,
        ):
            self.k = num_knn
            self.conv = "mr"
            self.act = "gelu"
            self.norm = "batch"
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
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["gnn_patch16_224"]
    return model


# ---------------------------------------------------------------
# Quick sanity check (optional)
# ---------------------------------------------------------------
if __name__ == "__main__":
    model = vig_ti_224_gelu()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
