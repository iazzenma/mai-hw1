#!/usr/bin/env python3
"""Create a Figure-6-style panel: masked input, reconstruction, and original
for different mask sampling strategies (random, block, grid).

Outputs: cls_outputs/mask_sampling_panel.png
"""
import argparse
import os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt

from model import MAE_ViT


def get_sample(val_dataset, idx=0):
    x, _ = val_dataset[idx]
    return x.unsqueeze(0)


def make_panel(model_random, model_block, model_grid, device='cuda'):
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    x = get_sample(val_dataset, idx=0).to(device)
    panels = []
    titles = ['random 75%', 'block 75%', 'grid 75%']
    for m in [model_random, model_block, model_grid]:
        with torch.no_grad():
            y, mask = m(x)
            masked_in = x * (1 - mask)
            recon = y * mask + x * (1 - mask)
        panels.append((masked_in, recon, x))

    # plot
    plt.figure(figsize=(12, 6))
    for col, (masked_in, recon, orig) in enumerate(panels):
        for row, img in enumerate([masked_in, recon, orig]):
            ax = plt.subplot(3, 3, col + 1 + 3*row)
            vis = (img[0].detach().cpu() * 0.5 + 0.5).clamp(0,1).permute(1,2,0).numpy()
            ax.imshow(vis)
            ax.axis('off')
            if row == 0:
                ax.set_title(titles[col])
    plt.tight_layout()
    os.makedirs('cls_outputs', exist_ok=True)
    out = 'cls_outputs/mask_sampling_panel.png'
    plt.savefig(out, dpi=200)
    print('Saved', out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_ckpt', type=str, required=True)
    parser.add_argument('--block_ckpt', type=str, required=True)
    parser.add_argument('--grid_ckpt', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mr, mb, mg = [torch.load(p, map_location='cpu').to(device) for p in [args.random_ckpt, args.block_ckpt, args.grid_ckpt]]
    for m in [mr, mb, mg]:
        m.eval()
    make_panel(mr, mb, mg, device=device)


if __name__ == '__main__':
    main()
