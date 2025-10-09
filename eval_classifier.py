#!/usr/bin/env python3
"""Evaluate a trained ViT_Classifier checkpoint on CIFAR-10 val set.

Usage:
  python3 eval_classifier.py --model_path vit-t-cls-ft-random.pt
  Optional: --batch_size 256 --out_json cls_outputs/ft_random_eval.json
"""
import argparse, json, os
import torch, torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--out_json', type=str, default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model_path, map_location='cpu').to(device)
    model.eval()

    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            losses.append(loss.item())
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.numel()

    avg_loss = float(np.mean(losses))
    acc = correct / total
    print(f'Validation: loss={avg_loss:.6f}, acc={acc:.6f} ({correct}/{total})')

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump({'model_path': args.model_path, 'val_loss': avg_loss, 'val_acc': acc,
                       'correct': correct, 'total': total}, f, indent=2)
        print('Wrote', args.out_json)


if __name__ == '__main__':
    main()
