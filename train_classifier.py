import os
import argparse
import math
import torch
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import torchvision.utils as vutils

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch.pt')
    parser.add_argument('--linear_probe', action='store_true', help='Freeze encoder and train only linear head')
    parser.add_argument('--metrics_path', type=str, default=None, help='Path to save metrics npz; defaults based on tag')
    parser.add_argument('--run_tag', type=str, default=None, help='Optional tag for logs/outputs')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        sub = 'pretrain-cls'
    else:
        model = MAE_ViT()
        sub = 'scratch-cls'
    model = ViT_Classifier(model.encoder, num_classes=10).to(device)

    # set up writer with optional tag
    tag = args.run_tag
    if tag is None:
        if args.pretrained_model_path is not None:
            base = os.path.splitext(os.path.basename(args.pretrained_model_path))[0]
        else:
            base = 'scratch'
        tag = f"{base}-{'lin' if args.linear_probe else 'ft'}"
    writer = SummaryWriter(os.path.join('logs', 'cifar10', sub, tag))

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    # linear probe: freeze encoder and optimize head only
    if args.linear_probe:
        for name, p in model.named_parameters():
            if not name.startswith('head.'):
                p.requires_grad = False
        params = model.head.parameters()
    else:
        params = model.parameters()

    optim = torch.optim.AdamW(params, lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    os.makedirs('cls_outputs', exist_ok=True)
    # derive metrics path if not provided
    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = os.path.join('cls_outputs', f"metrics_{tag}.npz")
    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    hist = { 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [] }
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
        hist['train_loss'].append(avg_train_loss)
        hist['train_acc'].append(avg_train_acc)

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  
            hist['val_loss'].append(avg_val_loss)
            hist['val_acc'].append(avg_val_acc)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            torch.save(model, args.output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)
        # persist metrics each epoch
        np.savez(metrics_path,
                 train_loss=np.array(hist['train_loss']), train_acc=np.array(hist['train_acc']),
                 val_loss=np.array(hist['val_loss']), val_acc=np.array(hist['val_acc']))

        # optional: save prediction grid for pretrained run
        if args.pretrained_model_path is not None and e in {0, args.total_epoch//2, args.total_epoch-1}:
            # take first 32 validation images and visualize predictions
            imgs = []
            gts = []
            preds = []
            with torch.no_grad():
                for i, (img, label) in enumerate(val_dataloader):
                    img = img.to(device)
                    logit = model(img)
                    pred = logit.argmax(dim=-1).detach().cpu()
                    imgs.append(img.detach().cpu())
                    gts.append(label)
                    preds.append(pred)
                    break
            img = torch.cat(imgs, dim=0)[:32]
            grid = vutils.make_grid(img, nrow=8, normalize=True, scale_each=True)
            vutils.save_image(grid, os.path.join('cls_outputs', f'pretrained_preds_epoch_{e:03d}.png'))