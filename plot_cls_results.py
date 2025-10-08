#!/usr/bin/env python3
"""Compare classification fine-tuning runs (scratch vs pretrained).

Expected files:
- cls_outputs/metrics_scratch.npz
- cls_outputs/metrics_pretrained.npz

Produces:
- cls_outputs/cls_loss_compare.png
- cls_outputs/cls_acc_compare.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curves(scratch, pretrained, key, outpath, ylabel, title):
    plt.figure(figsize=(8,5))
    if key in scratch:
        plt.plot(scratch[key], label=f'Scratch {key}')
    if key in pretrained:
        plt.plot(pretrained[key], label=f'Pretrained {key}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    print('Saved', outpath)


def main():
    os.makedirs('cls_outputs', exist_ok=True)
    s = np.load('cls_outputs/metrics_scratch.npz')
    p = np.load('cls_outputs/metrics_pretrained.npz')

    plot_curves(s, p, 'train_loss', 'cls_outputs/cls_train_loss_compare.png', 'Loss', 'Training Loss (Scratch vs Pretrained)')
    plot_curves(s, p, 'val_acc', 'cls_outputs/cls_val_acc_compare.png', 'Accuracy', 'Validation Accuracy (Scratch vs Pretrained)')


if __name__ == '__main__':
    main()
