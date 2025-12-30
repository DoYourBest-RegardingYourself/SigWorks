"""
SNN版本的特征学习训练脚本
使用SpikingJelly库训练脉冲神经网络
"""

import sys
import os
sys.path.append(".")
# 禁用输出缓冲，确保日志立即显示
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse
import pathlib
from collections import OrderedDict

import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchvision import transforms

import htcsignet.datasets.util as util
from htcsignet.feature_learning.data import TransformDataset

# SpikingJelly imports
from spikingjelly.activation_based import functional

from rich.progress import track
import htcsignet.feature_learning.models as models

@torch.compiler.disable
def save_parameters(params, dir, filename):
    torch.save(params, dir / filename)


def train(base_model: torch.nn.Module,
          classification_layer: torch.nn.Module,
          forg_layer: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          device: torch.device,
          args: Any,
          logdir: Optional[pathlib.Path]):
    """
    SNN训练函数
    
    注意: 每个batch前需要重置神经元状态
    """
    # Collect all parameters that need to be optimized
    parameters = list(base_model.parameters()) + list(classification_layer.parameters()) + list(forg_layer.parameters())

    # SNN通常使用Adam优化器
    optimizer = optim.Adam(parameters, lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    best_params = get_parameters(base_model, classification_layer, forg_layer)

    for epoch in track(range(args.epochs), description=f"Training SNN"):
        # Train one epoch; evaluate on validation
        train_epoch_snn(train_loader, base_model, classification_layer, forg_layer,
                        epoch, optimizer, lr_scheduler, device, args)

        val_metrics = test_snn(val_loader, base_model, classification_layer, device, args.forg, forg_layer)
        val_acc, val_loss, val_forg_acc, val_forg_loss = val_metrics

        # Save the best model only on improvement (early stopping)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_params = get_parameters(base_model, classification_layer, forg_layer)
            if logdir is not None:
                save_parameters(best_params, logdir, 'model_best_snn.pth')

        if args.forg:
            print('Epoch {}. Val loss: {:.4f}, Val acc: {:.2f}%, '
                  'Val forg loss: {:.4f}, Val forg acc: {:.2f}%'.format(epoch, val_loss,
                                                                        val_acc * 100,
                                                                        val_forg_loss,
                                                                        val_forg_acc * 100))
        else:
            print('Epoch {}. Val loss: {:.4f}, Val acc: {:.2f}%'.format(epoch, val_loss, val_acc * 100))

        if logdir is not None:
            current_params = get_parameters(base_model, classification_layer, forg_layer)
            save_parameters(current_params, logdir, 'model_last_snn.pth')

    return best_params


def copy_to_cpu(weights: Dict[str, Any]):
    return OrderedDict([(k, v.cpu()) for k, v in weights.items()])


def get_parameters(base_model, classification_layer, forg_layer):
    best_params = (copy_to_cpu(base_model.state_dict()),
                   copy_to_cpu(classification_layer.state_dict()),
                   copy_to_cpu(forg_layer.state_dict()))
    return best_params


def train_epoch_snn(train_loader: torch.utils.data.DataLoader,
                    base_model: torch.nn.Module,
                    classification_layer: torch.nn.Module,
                    forg_layer: torch.nn.Module,
                    epoch: int,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                    device: torch.device,
                    args: Any):
    """
    SNN单个epoch训练
    
    关键: 每个batch前需要调用 functional.reset_net() 重置神经元状态
    """
    base_model.train()
    classification_layer.train()
    forg_layer.train()
    
    step = 0
    n_steps = len(train_loader)
    total_loss = 0
    total_acc = 0
    
    for batch in track(train_loader, description=f"Epoch {epoch}"):
        x, y, yforg = batch[0], batch[1], batch[2]
        x = x.clone().float().to(device).detach()
        y = y.clone().long().to(device).detach()
        yforg = yforg.clone().float().to(device).detach()

        # 重要: 每个batch前重置SNN神经元状态
        functional.reset_net(base_model)

        # Forward propagation
        features = base_model(x)  # SNN输出已经是时间平均后的结果

        logits = classification_layer(features[yforg == 0])
        class_loss = F.cross_entropy(logits, y[yforg == 0])
        
        forg_logits = forg_layer(features).squeeze()
        forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)
        
        loss = (1 - args.lamb) * class_loss + args.lamb * forg_loss

        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 - 对于SNN训练很重要
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], args.grad_clip)

        # Update weights
        optimizer.step()

        pred = logits.argmax(1)
        label = y[yforg == 0]
        acc = label.eq(pred).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        step += 1
        
    lr_scheduler.step()
    
    avg_loss = total_loss / step
    avg_acc = total_acc / step
    print(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc*100:.2f}%')


def test_snn(val_loader: torch.utils.data.DataLoader,
             base_model: torch.nn.Module,
             classification_layer: torch.nn.Module,
             device: torch.device,
             is_forg: bool,
             forg_layer: Optional[torch.nn.Module] = None) -> Tuple[float, float, float, float]:
    """
    SNN验证函数
    """
    base_model.eval()
    classification_layer.eval()
    if forg_layer is not None:
        forg_layer.eval()

    val_losses = []
    val_accs = []
    val_forg_losses = []
    val_forg_accs = []
    
    for batch in track(val_loader, description=f"Validation"):
        x, y, yforg = batch[0], batch[1], batch[2]
        x = x.clone().float().to(device).detach()
        y = y.clone().long().to(device).detach()
        yforg = yforg.clone().float().to(device).detach()

        with torch.no_grad():
            # 重置神经元状态
            functional.reset_net(base_model)
            
            features = base_model(x)
            logits = classification_layer(features[yforg == 0])

            loss = F.cross_entropy(logits, y[yforg == 0])
            pred = logits.argmax(1)
            acc = y[yforg == 0].eq(pred).float().mean()

            if is_forg and forg_layer is not None:
                forg_logits = forg_layer(features).squeeze()
                forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)
                forg_pred = forg_logits > 0
                forg_acc = yforg.long().eq(forg_pred.long()).float().mean()

                val_forg_losses.append(forg_loss.item())
                val_forg_accs.append(forg_acc.item())

        val_losses.append(loss.item())
        val_accs.append(acc.item())
        
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_forg_loss = np.mean(val_forg_losses) if len(val_forg_losses) > 0 else np.nan
    val_forg_acc = np.mean(val_forg_accs) if len(val_forg_accs) > 0 else np.nan

    return val_acc, val_loss, val_forg_acc, val_forg_loss


def apply_random(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set all random seed to {seed}')


def main(args):
    logdir = pathlib.Path(args.model_checkpoint)
    if not logdir.exists():
        logdir.mkdir()

    def get_device():
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            raise RuntimeError('No GPU available')
        return device

    device = get_device()
    print('Using device: {}'.format(device))

    apply_random(args.seed)

    print('Loading Data')
    sys.stdout.flush()

    x, y, yforg, usermapping, filenames = util.load_dataset(args.dataset_path)
    print(f'Dataset loaded: {x.shape[0]} samples')
    sys.stdout.flush()
    
    data = util.get_subset((x, y, yforg), subset=range(*args.users))
    print(f'Subset selected: {data[0].shape[0]} samples')
    sys.stdout.flush()
    
    if not args.forg:
        data = util.remove_forgeries(data, forg_idx=2)

    train_loader, val_loader = setup_data_loaders(data, args.batch_size, args.input_size)
    print(f'Data loaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches')
    sys.stdout.flush()

    print('Initializing SNN Model')
    sys.stdout.flush()

    n_classes = len(np.unique(data[1]))
    
    if '_snn' not in args.model:
        raise ValueError('Please select an SNN model for SNN training.')
    base_model = models.available_models[args.model](args.weights).to(device)
    # 注意: 不使用 torch.compile，因为 SpikingJelly 与 dynamo 不兼容
    # base_model = torch.compile(base_model)
    
    # 分类层和伪造检测层
    classification_layer = nn.Linear(1280, n_classes).to(device)
    forg_layer = nn.Linear(1280, 1).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f'SNN Model Parameters: {total_params / 1e6:.2f}M')
    print(f'Timesteps (T): {args.timesteps}')
    print(f'Tau: {args.tau}')

    print('Training SNN')
    train(base_model, classification_layer, forg_layer, train_loader, val_loader,
          device, args, logdir)


def setup_data_loaders(data, batch_size, input_size):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[1])
    data = TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(y), torch.from_numpy(data[2]))
    train_size = int(0.9 * len(data))
    sizes = (train_size, len(data) - train_size)
    train_set, test_set = random_split(data, sizes)
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
    ])
    train_set = TransformDataset(train_set, train_transforms)
    
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    test_set = TransformDataset(test_set, val_transforms)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Train SNN Signet/F')
    
    # 数据相关参数
    argparser.add_argument('--dataset-path', help='Path containing a numpy file with images and labels', 
                          default='./datasets/GPDS_1000_256X256.npz')
    argparser.add_argument('--input-size', help='Input size (cropped)', nargs=2, type=int, default=(224, 224))
    argparser.add_argument('--users', nargs=2, type=int, default=(300, 1000))

    # 模型相关参数
    argparser.add_argument('--model', default='vig_snn', 
                          choices=['vig_snn',])
    argparser.add_argument('--batch-size', help='Batch size', type=int, default=16)  # SNN通常用更小的batch size

    # SNN特有参数
    argparser.add_argument('--timesteps', '-T', help='Number of timesteps for SNN', type=int, default=4)
    argparser.add_argument('--tau', help='Membrane time constant for LIF neuron', type=float, default=2.0)
    argparser.add_argument('--grad-clip', help='Gradient clipping value', type=float, default=1.0)

    # 学习率相关参数
    argparser.add_argument('--lr', help='learning rate', default=1e-4, type=float)  # SNN通常需要更大的学习率
    argparser.add_argument('--lr-decay', help='learning rate decay (multiplier)', default=0.1, type=float)
    argparser.add_argument('--lr-decay-times', help='number of times learning rate decays', default=5, type=float)
    argparser.add_argument('--momentum', help='momentum', default=0.90, type=float)
    argparser.add_argument('--weight-decay', help='Weight Decay', default=1e-4, type=float)
    argparser.add_argument('--epochs', help='Number of epochs', default=100, type=int)

    # 其他参数
    argparser.add_argument('--seed', default=42, type=int)
    argparser.add_argument('--lamb', type=float, default=0.95, help='Lambda for loss weighting')
    argparser.add_argument('--model_checkpoint', help='model checkpoint', default='./model_checkpoint_snn/')
    argparser.add_argument('--forg', type=bool, help='Train with forgeries detection task', default=True)
    
    arguments = argparser.parse_args()
    print("=" * 60)
    print("SNN Training Configuration")
    print("=" * 60)
    print(arguments)
    print("=" * 60)

    main(arguments)
