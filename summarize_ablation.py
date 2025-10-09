#!/usr/bin/env python3
"""Summarize ablation runs into a markdown table and grouped bar chart.

Supports combining FT accuracies (from JSON eval files) and LIN best accuracies
(from metrics_<tag>.npz produced during linear-probe training).

Examples:
  python3 summarize_ablation.py \
    --ft_json random:cls_outputs/ft_random_eval.json \
    --ft_json block75:cls_outputs/ft_block75_eval.json \
    --ft_json grid75:cls_outputs/ft_grid75_eval.json \
    --ft_json encmask:cls_outputs/ft_encmask_eval.json \
    --lin_tag random:vit-t-mae-sample-random-150ep-lin \
    --lin_tag block75:vit-t-mae-sample-block75-150ep-lin \
    --lin_tag grid75:vit-t-mae-sample-grid75-150ep-lin \
    --lin_tag encmask:vit-t-mae-encmask-150ep-lin \
    --out_md cls_outputs/ablation_ft_lin.md \
    --out_png cls_outputs/ablation_grouped_bars.png
"""
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt


def load_lin_best(tag):
    path = os.path.join('cls_outputs', f'metrics_{tag}.npz')
    m = np.load(path)
    return float(m['val_acc'].max()), int(m['val_acc'].argmax())


def load_ft_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    return float(d['val_acc'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ft_json', action='append', default=[], help='variant:path_to_ft_json (val_acc)')
    ap.add_argument('--lin_tag', action='append', default=[], help='variant:metrics_tag (reads cls_outputs/metrics_<tag>.npz)')
    ap.add_argument('--out_md', default='cls_outputs/ablation_ft_lin.md')
    ap.add_argument('--out_png', default='cls_outputs/ablation_grouped_bars.png')
    args = ap.parse_args()

    os.makedirs('cls_outputs', exist_ok=True)

    ft = {}
    for item in args.ft_json:
        var, path = item.split(':', 1)
        ft[var] = load_ft_json(path)

    lin = {}
    linepoch = {}
    for item in args.lin_tag:
        var, tag = item.split(':', 1)
        acc, ep = load_lin_best(tag)
        lin[var] = acc
        linepoch[var] = ep

    variants = sorted(set(ft.keys()) | set(lin.keys()))

    # markdown table
    with open(args.out_md, 'w') as f:
        f.write('| Variant | FT acc | LIN best acc | LIN best epoch |\n')
        f.write('|---|---:|---:|---:|\n')
        for v in variants:
            fta = ft.get(v, float('nan'))
            lina = lin.get(v, float('nan'))
            epa = linepoch.get(v, -1)
            f.write(f'| {v} | {fta:.4f} | {lina:.4f} | {epa} |\n')
    print('Wrote', args.out_md)

    # grouped bars
    x = np.arange(len(variants))
    width = 0.38
    ft_vals = [ft.get(v, np.nan) for v in variants]
    lin_vals = [lin.get(v, np.nan) for v in variants]

    plt.figure(figsize=(max(6, len(variants)*1.6), 4.5))
    plt.bar(x - width/2, ft_vals, width, label='FT')
    plt.bar(x + width/2, lin_vals, width, label='LIN')
    plt.xticks(x, variants, rotation=10)
    plt.ylabel('Accuracy (val)')
    plt.title('Ablation: FT vs LIN per variant')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print('Saved', args.out_png)


if __name__ == '__main__':
    main()
