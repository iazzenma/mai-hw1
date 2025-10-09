# MAE Pre-Training on CIFAR-10

## Codebase overview

- `mae_pretrain.py` — MAE pretraining on CIFAR-10. Logs to TensorBoard, saves `losses.npy`, and writes recon visualizations to `outputs/`.
- `model.py` — MAE with ViT encoder/decoder.
- `utils.py` — Utilities (e.g., `setup_seed`).
- `train_classifier.py` — Optional downstream classifier training.
- `plot_results.py` — Plots loss curve and prepares best/last reconstruction images.

## Environment

- Python: 3.9.23 (main, Jun  5 2025, 13:40:20)  [GCC 11.2.0]
- PyTorch: 2.5.1
- CUDA available: True
- GPU(s): 4 × NVIDIA GeForce RTX 4090 (devices 0–3)

## Method

- Dataset: CIFAR-10 (train split for pretraining; val split used for visualization).
- Model: MAE-ViT (tiny) with masking ratio `--mask_ratio` (default 0.75).
- Objective: Minimize MSE over the masked regions: E[(x̂ − x)^2 · mask] / mask_ratio.
- Optimization: AdamW with base LR scaled by batch size: `base_lr * (batch_size / 256)` and cosine schedule with warmup.
- Batch accumulation: `batch_size` split into chunks of `max_device_batch_size` to match device memory; gradients accumulated across steps.
- Logging: TensorBoard scalars for loss; per-epoch visualization grids saved to `outputs/`.

CLI example used (150 epochs):

```bash
CUDA_VISIBLE_DEVICES=0 python3 mae_pretrain.py \
  --batch_size 512 --max_device_batch_size 256 \
  --total_epoch 150 --model_path vit-t-mae-150ep.pt
```

## Results

### Training loss curve

The MAE loss across epochs is shown below:

![MAE Loss Curve](outputs/loss_curve.png)

Stats:
- Epochs: 150
- First 5 losses: [0.2381, 0.1851, 0.1808, 0.1774, 0.1723]
- Last 5 losses: [0.04026, 0.04019, 0.04014, 0.04021, 0.04015]
- Best loss: 0.04014 at epoch 147
- Final loss: 0.04015

### Reconstruction visualizations

Each visualization shows three rows (top to bottom): masked input, model reconstruction, and original image. The grid stacks 16 images in a 2×8 layout.

- Best epoch (by min loss):

![Best Reconstruction](outputs/best_recon.png)

- Final epoch:

![Last Reconstruction](outputs/last_recon.png)

## Discussion and insights

- Learning dynamics: With 75% masking, the model focuses on global structure; loss typically drops quickly in the first 20–40 epochs, then tapers under cosine decay.
- Qualitative fidelity: Reconstructions should capture coarse shapes and colors; fine textures and small object details are often smoothed due to heavy masking and decoder capacity.
- Mask ratio effect: Higher ratios make the task harder but encourage stronger encoder representations; on CIFAR-10 resolution, 0.75 is a good balance.
- Overfitting: MAE’s objective reconstructs input pixels; overfitting manifests as diminishing improvements despite training loss decreasing slowly. Monitoring downstream validation (classifier fine-tune) is recommended for representation quality.
- Compute considerations: Gradient accumulation enables larger effective batch sizes on limited memory; ensure `batch_size % max_device_batch_size == 0`.

## Conclusion

- You completed 150-epoch MAE pretraining on CIFAR-10 and produced a stable decreasing loss curve and reasonable reconstructions.
- The encoder from this run can now be fine-tuned for classification using `train_classifier.py` to quantify representation gains over training from scratch.

## Reproducibility

- Seed set with `--seed` (default 42).
- Command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 mae_pretrain.py --batch_size 512 --max_device_batch_size 256 --total_epoch 150 --model_path vit-t-mae-150ep.pt
```

- Artifacts: `losses.npy`, `outputs/` (images including `loss_curve.png`, `best_recon.png`, `last_recon.png`), model checkpoint `vit-t-mae-150ep.pt`.

---

# Phase 2 — Classification Fine-Tuning on CIFAR-10

## Setup

- Objective: Compare classification performance when fine-tuning ViT-T with and without MAE pretraining.
- Dataset: CIFAR-10 (train/val splits from torchvision).
- Model: `ViT_Classifier` built from the MAE encoder; classifier head trained from scratch in both cases.
- Optimization: AdamW, cosine LR with warmup; batch accumulation supported.
- Epochs: ≥ 50 for each run (scratch and pretrained).



Commands used:

```bash
# Scratch (50 epochs)
CUDA_VISIBLE_DEVICES=0 python3 train_classifier.py \
  --total_epoch 50 \
  --batch_size 128 --max_device_batch_size 128 \
  --output_model_path vit-t-classifier-from_scratch.pt

# Pretrained (50 epochs)
CUDA_VISIBLE_DEVICES=0 python3 train_classifier.py \
  --total_epoch 50 \
  --batch_size 128 --max_device_batch_size 128 \
  --pretrained_model_path vit-t-mae-150ep.pt \
  --output_model_path vit-t-classifier-from_pretrained.pt
```

## Results

### 1) Training loss curve comparison

![CLS Training Loss Comparison](cls_outputs/cls_train_loss_compare.png)

Observation: The pretrained model starts with substantially lower loss and converges faster, indicating better initialization from MAE.

### 2) Validation accuracy curve comparison

![CLS Validation Accuracy Comparison](cls_outputs/cls_val_acc_compare.png)

Observation: The pretrained model reaches higher validation accuracy and does so in fewer epochs (improved sample efficiency). The scratch model trails but continues to improve more slowly.

Summary metrics:
- Scratch: best val acc = 0.7304 at epoch 43 (of 50)
- Pretrained: best val acc = 0.8222 at epoch 49 (of 50)

### 3) Visualization of classifier predictions (MAE-pretrained)

Below is a grid of validation images with predicted (P) vs true (T) labels; green titles indicate correct predictions, red indicate errors.

![Pretrained Classifier Predictions](cls_outputs/pretrained_preds_grid.png)

Note: Additional epoch snapshots were saved during training as `cls_outputs/pretrained_preds_epoch_*.png` (epoch 0, mid, and last) to illustrate progression.

## Discussion & insights

- Pretraining benefit: MAE-pretrained encoder significantly accelerates convergence and boosts final accuracy relative to training from scratch, consistent with self-supervised pretraining literature.
- Sample efficiency: With the same number of epochs and batch size, the pretrained model achieves higher accuracy earlier, reducing the compute needed to reach a target accuracy.
- Error patterns: Misclassifications tend to occur among visually similar classes (e.g., cat/dog, airplane/ship at low resolution). Class-balanced augmentation or longer fine-tuning can further close the gap.
- Head vs encoder: Only the classifier head is randomly initialized; the encoder benefits from rich representations learned under heavy masking (0.75), which favor global structure—useful for CIFAR-10.

## Phase-2 conclusion

Fine-tuning the classifier initialized from an MAE-pretrained encoder yields better and faster results than training from scratch. The curves and prediction visualizations support improved convergence, higher validation accuracy, and more reliable predictions.

---

# Phase 3 — Ablation Study

We follow Table 1 ablations (c) Mask token and (f) Mask sampling. All ablations are conducted in the MAE pre-training phase, followed by classification fine-tuning for 50 epochs using the same protocol as Phase 2.

## Ablations implemented

- (c) Mask token in encoder
  - New flag: `--encoder_with_mask_token`
  - Behavior: The encoder explicitly inserts mask tokens at masked patch positions (default is without encoder mask tokens). The decoder always has its own mask token.

- (f) Mask sampling strategy in encoder
  - New flag: `--mask_sampling {random, block, grid}`
  - Variants to run (75% mask):
    - random (baseline)
    - block 75 (contiguous square region masked)
    - grid 75 (regular grid keeps ~25% visible)

## How to run (pretraining)

```bash
# (c) Mask token ON
python3 mae_pretrain.py --total_epoch 150 --mask_sampling random --encoder_with_mask_token \
  --model_path vit-t-mae-encmask-150ep.pt

# (f) Sampling variants
python3 mae_pretrain.py --total_epoch 150 --mask_sampling random --model_path vit-t-mae-sample-random-150ep.pt
python3 mae_pretrain.py --total_epoch 150 --mask_sampling block  --model_path vit-t-mae-sample-block75-150ep.pt
python3 mae_pretrain.py --total_epoch 150 --mask_sampling grid   --model_path vit-t-mae-sample-grid75-150ep.pt
```

Then fine-tune each encoder checkpoint for classification (50 epochs):

```bash
python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-encmask-150ep.pt \
  --output_model_path vit-t-cls-encmask.pt

python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-sample-random-150ep.pt \
  --output_model_path vit-t-cls-sample-random.pt
python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-sample-block75-150ep.pt \
  --output_model_path vit-t-cls-sample-block75.pt
python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-sample-grid75-150ep.pt \
  --output_model_path vit-t-cls-sample-grid75.pt
```

## Expected observations (from MAE paper and our setup)

- (c) Mask token: Removing mask tokens from the encoder tends to be faster and slightly more accurate; enabling encoder mask tokens may reduce speed and not help accuracy in ViT, but the effect can be dataset/scale dependent.
- (f) Sampling: Random masking is typically best; block masking at the same ratio can degrade performance; a regular grid keeps evenly spaced visible patches and may be close to random but usually slightly worse.

## Results and analysis

Please insert your measured curves and final validation accuracies for each variant. Recommended artifacts:
- Pretraining loss curves per variant (optional).
- Fine-tuning curves: training loss and validation accuracy comparisons across variants.
- A small table summarizing best validation accuracy per variant.

Avg forward/no-enc-mask-token: 13.17 ms; with-enc-mask-token: 14.87 ms

Discussion points:
- How mask tokens in the encoder affect representation learning and downstream accuracy.
- How spatial correlation in the masking pattern (block/grid) impacts reconstruction and the learned encoder features compared to random masking.

### Measured results (our runs)

#### (c) Mask token (encoder)

| Variant | FT acc | LIN best acc | LIN best epoch |
|---|---:|---:|---:|
| encoder w/ mask token | 0.7756 | 0.3851 | 38 |
| encoder w/o mask token (baseline, random 75) | 0.8223 | 0.5311 | 44 |

Figure for (c): `cls_outputs/enc_mask_token_recon.png`

Observation: Enabling encoder mask tokens reduced both FT and LIN accuracies vs. baseline, aligning with the paper’s conclusion that an encoder without mask tokens is more accurate and faster.

#### (f) Mask sampling at 75% mask ratio

| Variant | FT acc | LIN best acc | LIN best epoch |
|---|---:|---:|---:|
| random 75 | 0.8223 | 0.5311 | 44 |
| block 75 | 0.8093 | 0.5722 | 43 |
| grid 75 | 0.7908 | 0.4822 | 33 |

Figure for (f): `cls_outputs/mask_sampling_panel.png`

Observation: FT accuracy follows random > block > grid. LIN shows block > random > grid in our runs—on CIFAR-10 with a small ViT and linear head, block masking may yield features that are slightly more linearly separable. Random remains a robust default.

### Combined FT/LIN grouped summary

We generated a grouped bar chart comparing FT and LIN accuracies across variants:

![Ablation FT/LIN Grouped Bars](cls_outputs/ablation_grouped_bars.png)

For the exact numbers used in this figure, see `cls_outputs/ablation_ft_lin.md`.

Context: Phase-2 runs on this setup achieved pretrained FT best=0.8222 @49 vs scratch FT best=0.7304 @43.
