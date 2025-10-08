#!/usr/bin/env python3
"""Plot training loss and pick reconstruction images.

Usage: python3 plot_results.py

Assumes `losses.npy` exists in the current directory and reconstructions are saved as
`outputs/recon_epoch_###.png` by `mae_pretrain.py`.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():
    if not os.path.exists('losses.npy'):
        print('losses.npy not found in current directory.')
        return

    losses = np.load('losses.npy')
    epochs = np.arange(len(losses))

    plt.figure(figsize=(8,5))
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve of MAE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/loss_curve.png')
    print('Saved outputs/loss_curve.png')

    # best_idx = int(np.argmin(losses))
    # last_idx = len(losses) - 1

    # best_fname = os.path.join('outputs', f'recon_epoch_{best_idx:03d}.png')
    # last_fname = os.path.join('outputs', f'recon_epoch_{last_idx:03d}.png')

    # if os.path.exists(best_fname):
    #     Image.open(best_fname).save(os.path.join('outputs', 'best_recon.png'))
    #     print(f'Saved best reconstruction: outputs/best_recon.png (epoch {best_idx})')
    # else:
    #     print(f'Best reconstruction file not found: {best_fname}')

    # if os.path.exists(last_fname):
    #     Image.open(last_fname).save(os.path.join('outputs', 'last_recon.png'))
    #     print(f'Saved last reconstruction: outputs/last_recon.png (epoch {last_idx})')
    # else:
    #     print(f'Last reconstruction file not found: {last_fname}')


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    main()
