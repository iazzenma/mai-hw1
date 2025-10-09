#!/usr/bin/env python3
"""Showcase (c) Mask token: compare with vs without encoder mask tokens.

Outputs:
- cls_outputs/enc_mask_token_recon.png (masked input, recon, original for both variants)
- Prints average forward time per pass for each variant.
"""
import argparse
import os
import time
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt


def get_sample(val_dataset, idx=0):
    x, _ = val_dataset[idx]
    return x.unsqueeze(0)


def time_forward(model, x, n=50):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(n):
            _ = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
    return (t1 - t0) / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noenc_ckpt', type=str, required=True, help='Checkpoint WITHOUT encoder mask tokens')
    parser.add_argument('--enc_ckpt', type=str, required=True, help='Checkpoint WITH encoder mask tokens')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    x = get_sample(val_dataset, args.idx).to(device)

    m_noenc = torch.load(args.noenc_ckpt, map_location='cpu').to(device)
    m_enc = torch.load(args.enc_ckpt, map_location='cpu').to(device)
    m_noenc.eval(); m_enc.eval()

    with torch.no_grad():
        y0, mask0 = m_noenc(x)
        y1, mask1 = m_enc(x)
        masked0 = x * (1 - mask0); recon0 = y0 * mask0 + x * (1 - mask0)
        masked1 = x * (1 - mask1); recon1 = y1 * mask1 + x * (1 - mask1)

    t0 = time_forward(m_noenc, x)
    t1 = time_forward(m_enc, x)
    print(f'Avg forward/no-enc-mask-token: {t0*1000:.2f} ms; with-enc-mask-token: {t1*1000:.2f} ms')

    def denorm(img):
        return (img.detach().cpu() * 0.5 + 0.5).clamp(0,1)

    plt.figure(figsize=(8,6))
    panels = [
        ('no-enc-mask: masked', denorm(masked0[0])),
        ('no-enc-mask: recon', denorm(recon0[0])),
        ('no-enc-mask: orig', denorm(x[0])),
        ('enc-mask: masked', denorm(masked1[0])),
        ('enc-mask: recon', denorm(recon1[0])),
        ('enc-mask: orig', denorm(x[0])),
    ]
    for i, (title, img) in enumerate(panels):
        ax = plt.subplot(2, 3, i+1)
        ax.imshow(img.permute(1,2,0).numpy())
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs('cls_outputs', exist_ok=True)
    out = 'cls_outputs/enc_mask_token_recon.png'
    plt.savefig(out, dpi=200)
    print('Saved', out)


if __name__ == '__main__':
    main()
