import sys

sys.path.append(".")
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
import htcsignet.feature_learning.models as models
# import os

from rich.progress import track


def train(base_model: torch.nn.Module,
          classification_layer: torch.nn.Module,
          forg_layer: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          device: torch.device,
          args: Any,
          logdir: Optional[pathlib.Path]):
    # Collect all parameters that need to be optimizer
    # parameters = list(base_model.parameters()) + list(classification_layer.parameters())
    parameters = list(base_model.parameters()) + list(classification_layer.parameters()) + list(forg_layer.parameters())

    # Initialize optimizer and learning rate scheduler
    # optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
    #                       nesterov=True, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.Adam(parameters, lr=args.lr)  # VIT中使用Adam优化器最好的学习率是1e-5
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             args.epochs // args.lr_decay_times,
                                             args.lr_decay)

    best_acc = 0
    best_params = get_parameters(base_model, classification_layer, forg_layer)

    for epoch in track(range(args.epochs), description=f"Run train"):
        # Train one epoch; evaluate on validation
        train_epoch(train_loader, base_model, classification_layer, forg_layer,
                    epoch, optimizer, lr_scheduler, device, args)

        val_metrics = test(val_loader, base_model, classification_layer, device, args.forg, forg_layer)
        val_acc, val_loss, val_forg_acc, val_forg_loss = val_metrics

        # Save the best model only on improvement (early stopping)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_params = get_parameters(base_model, classification_layer, forg_layer)
            if logdir is not None:
                torch.save(best_params, logdir / 'model_best.pth')

        if args.forg:
            print('Epoch {}. Val loss: {:.4f}, Val acc: {:.2f}%,'
                  'Val forg loss: {:.4f}, Val forg acc: {:.2f}%'.format(epoch, val_loss,
                                                                        val_acc * 100,
                                                                        val_forg_loss,
                                                                        val_forg_acc * 100))
        else:
            print('Epoch {}. Val loss: {:.4f}, Val acc: {:.2f}%'.format(epoch, val_loss, val_acc * 100))

        if logdir is not None:
            current_params = get_parameters(base_model, classification_layer, forg_layer)
            torch.save(current_params, logdir / 'model_last.pth')

    return best_params


def copy_to_cpu(weights: Dict[str, Any]):
    return OrderedDict([(k, v.cpu()) for k, v in weights.items()])


def get_parameters(base_model, classification_layer, forg_layer):
    best_params = (copy_to_cpu(base_model.state_dict()),
                   copy_to_cpu(classification_layer.state_dict()),
                   copy_to_cpu(forg_layer.state_dict()))
    return best_params


def train_epoch(train_loader: torch.utils.data.DataLoader,
                base_model: torch.nn.Module,
                classification_layer: torch.nn.Module,
                forg_layer: torch.nn.Module,
                epoch: int,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                args: Any):
    base_model.train()
    step = 0
    n_steps = len(train_loader)
    for batch in track(train_loader, description=f"Run batch"):
        x, y, yforg = batch[0], batch[1], batch[2]
        x = x.clone().float().to(device).detach()
        y = y.clone().long().to(device).detach()
        yforg = yforg.clone().float().to(device).detach()

        # Forward propagation
        features = base_model(x)

    
        logits = classification_layer(features[yforg == 0])
        class_loss = F.cross_entropy(logits, y[yforg == 0])
        forg_logits = forg_layer(features).squeeze()
        forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)
        loss = (1 - args.lamb) * class_loss
        loss += args.lamb * forg_loss
        
			
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'], 10)

        # Update weights
        optimizer.step()

        pred = logits.argmax(1)

        label = y[yforg == 0]
        acc = label.eq(pred).float().mean()

        step += 1
    lr_scheduler.step()


def test(val_loader: torch.utils.data.DataLoader,
         base_model: torch.nn.Module,
         classification_layer: torch.nn.Module,
         device: torch.device,
         is_forg: bool,
         forg_layer: Optional[torch.nn.Module] = None) -> Tuple[float, float, float, float]:
    base_model.eval()

    val_losses = []
    val_accs = []

    val_forg_losses = []
    val_forg_accs = []
    for batch in track(val_loader, description=f"Run validation"):
        x, y, yforg = batch[0], batch[1], batch[2]
        x = x.clone().float().to(device).detach()
        y = y.clone().long().to(device).detach()
        yforg = yforg.clone().float().to(device).detach()

        with torch.no_grad():
            features = base_model(x)
            logits = classification_layer(features[yforg == 0])

            # logits = base_model(x)[yforg == 0]
            loss = F.cross_entropy(logits, y[yforg == 0])
            pred = logits.argmax(1)
            acc = y[yforg == 0].eq(pred).float().mean()

            if is_forg:
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

    x, y, yforg, usermapping, filenames = util.load_dataset(args.dataset_path)
    data = util.get_subset((x, y, yforg), subset=range(*args.users))
    if not args.forg:
        data = util.remove_forgeries(data, forg_idx=2)

    train_loader, val_loader = setup_data_loaders(data, args.batch_size, args.input_size)

    print('Initializing Model')

    
    n_classes = len(np.unique(data[1]))
    base_model = models.available_models[args.model](args.weights).to(device)
    #base_model = models.available_models[args.model](args.weights)
    #base_model = torch.nn.DataParallel(base_model, device_ids=[0, 1]).cuda()
    classification_layer = nn.Linear(1280, n_classes).to(device)
    # classification_layer = torch.nn.DataParallel(classification_layer, device_ids=[0, 1, 2]).cuda()
    forg_layer = nn.Linear(1280, 1).to(device)
    # forg_layer = torch.nn.DataParallel(forg_layer, device_ids=[0, 1, 2]).cuda()

   
    print('Training')
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Train Signet/F')
    argparser.add_argument('--dataset-path', help='Path containing a numpy file with images and labels', default='./datasets/GPDS_1000_256X256.npz')
    argparser.add_argument('--input-size', help='Input size (cropped)', nargs=2, type=int, default=(224, 224))
    argparser.add_argument('--users', nargs=2, type=int, default=(300, 1000))

    argparser.add_argument('--model', default='htcsignet', choices=models.available_models)
    argparser.add_argument('--batch-size', help='Batch size', type=int, default=32)

    argparser.add_argument('--lr-decay', help='learning rate decay (multiplier)', default=0.1, type=float)
    argparser.add_argument('--lr-decay-times', help='number of times learning rate decays', default=5, type=float)
    argparser.add_argument('--momentum', help='momentum', default=0.90, type=float)
    argparser.add_argument('--weight-decay', help='Weight Decay', default=1e-4, type=float)
    argparser.add_argument('--epochs', help='Number of epochs', default=100, type=int)

    argparser.add_argument('--seed', default=0, type=int) #42

    argparser.add_argument('--lamb', type=float, default=0.95)

    argparser.add_argument('--weights', default='./vit_base_patch16_224.pth') #'../../vit_base_patch16_224.pth'
    argparser.add_argument('--freeze_layers', default='True')
    argparser.add_argument('--model_checkpoint', help='model checkpoint', default='./model_checkpoint/')
    argparser.add_argument('--lr', help='learning rate', default=1e-5, type=float)
    argparser.add_argument('--forg', type=bool, help='Train with forgeries detection task', default=True)
    arguments = argparser.parse_args()
    print(arguments)

    main(arguments)
