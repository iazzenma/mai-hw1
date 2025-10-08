# MAE Pretraining on CIFAR-10 — Report Template

This document describes the codebase, environment, running instructions for MAE pretraining on CIFAR-10, and includes placeholders for figures and results you will generate by running the training script locally.

## 1. Codebase structure

- `mae_pretrain.py`: Main pretraining script for MAE on CIFAR-10. Saves `losses.npy` and reconstruction images to `outputs/`.
- `model.py`: MAE + ViT model definition (encoder/decoder). (See repository for details.)
- `utils.py`: Utility functions (seed setup, etc.).
- `train_classifier.py`: Script to train a downstream classifier (not modified here).
- `plot_results.py`: Utility script that plots `losses.npy` and copies best/last reconstructions to `outputs/`.
- `requirements.txt`: Python package requirements.

## 2. Environment

Run these commands on your machine to collect environment information (Python, PyTorch, CUDA, GPU):

```bash
python3 - << 'PY'
import sys, torch
print('python_version:', sys.version.replace('\n',' '))
print('torch_version:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print('device', i, torch.cuda.get_device_name(i))
else:
    print('no cuda')
PY
```

Paste the command output here after running.

## 3. How to run MAE pretraining (150 epochs)

Example command (recommended if you have a single GPU):

```bash
python3 mae_pretrain.py --batch_size 512 --max_device_batch_size 256 --total_epoch 150 --model_path vit-t-mae-150ep.pt
```

Notes:
- Adjust `--batch_size` to match your GPU memory. The script divides `batch_size` by `max_device_batch_size` to support gradient accumulation.
- The script will save per-epoch loss in `losses.npy` and reconstructions to `outputs/recon_epoch_###.png`.

If you want to do a quick smoke test first (10 epochs):

```bash
python3 mae_pretrain.py --batch_size 256 --max_device_batch_size 128 --total_epoch 10 --model_path vit-t-mae-smoke.pt
```

## 4. Plotting and collecting results

After training completes, run:

```bash
python3 plot_results.py
```

This will create `outputs/loss_curve.png`, `outputs/best_recon.png` and `outputs/last_recon.png`.

## 5. Report placeholders

- Environment info: (paste output from step 2)
- Loss curve: `outputs/loss_curve.png`
- Best reconstruction: `outputs/best_recon.png` (best epoch by lowest average loss)
- Final reconstruction: `outputs/last_recon.png`

## 6. Suggested analysis and discussion points

- Describe training hyperparameters used (batch size, mask ratio, learning rate, epochs, weight decay).
- Present the loss curve and discuss convergence speed and whether 150 epochs was sufficient.
- Show reconstruction images in a 3-column format: masked input, model reconstruction, original image. Comment on qualitative fidelity, artifacts, and how masking ratio affects results.
- If possible, quantify reconstruction error per-channel or patch-wise, and note where the model struggles (fine texture, edges, color fidelity).

## 7. Next steps

- Fine-tune on downstream task (classification) using encoder weights.
- Run with different mask ratios or model sizes to measure effect.
