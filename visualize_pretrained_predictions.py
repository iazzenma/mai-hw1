#!/usr/bin/env python3
"""Visualize classifier predictions for the pretrained model on CIFAR-10 val set.

Outputs: cls_outputs/pretrained_preds_grid.png
"""
import argparse
import os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='vit-t-classifier-from_pretrained.pt')
    parser.add_argument('--num_images', type=int, default=32)
    parser.add_argument('--nrow', type=int, default=8)
    parser.add_argument('--out', type=str, default='cls_outputs/pretrained_preds_grid.png')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model_path, map_location='cpu').to(device)
    model.eval()

    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    imgs, labels = [], []
    for i in range(args.num_images):
        x, y = val_dataset[i]
        imgs.append(x)
        labels.append(y)
    x = torch.stack(imgs, dim=0).to(device)
    y = torch.tensor(labels)

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=-1).cpu()

    # denormalize for display
    x_vis = (x.detach().cpu() * 0.5 + 0.5).clamp(0,1)

    cols = args.nrow
    rows = (args.num_images + cols - 1)//cols
    plt.figure(figsize=(cols*2, rows*2))
    for idx in range(args.num_images):
        ax = plt.subplot(rows, cols, idx+1)
        img = x_vis[idx].permute(1,2,0).numpy()
        ax.imshow(img)
        pred_name = CLASS_NAMES[int(preds[idx])]
        true_name = CLASS_NAMES[int(y[idx])]
        color = 'g' if preds[idx]==y[idx] else 'r'
        ax.set_title(f"P:{pred_name}\nT:{true_name}", color=color, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
