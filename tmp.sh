#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# # (c) Mask token
# python3 mae_pretrain.py --total_epoch 150 --mask_sampling random --encoder_with_mask_token \
#   --model_path vit-t-mae-encmask-150ep.pt --batch_size 512 --max_device_batch_size 256

# python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-encmask-150ep.pt \
#   --output_model_path vit-t-cls-encmask.pt --batch_size 128 --max_device_batch_size 128

# # (f) Mask sampling variants
# python3 mae_pretrain.py --total_epoch 150 --mask_sampling random --model_path vit-t-mae-sample-random-150ep.pt --batch_size 512 --max_device_batch_size 256
# python3 mae_pretrain.py --total_epoch 150 --mask_sampling block  --model_path vit-t-mae-sample-block75-150ep.pt --batch_size 512 --max_device_batch_size 256
# python3 mae_pretrain.py --total_epoch 150 --mask_sampling grid   --model_path vit-t-mae-sample-grid75-150ep.pt --batch_size 512 --max_device_batch_size 256

# python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-sample-random-150ep.pt \
#   --output_model_path vit-t-cls-sample-random.pt --batch_size 128 --max_device_batch_size 128
# python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-sample-block75-150ep.pt \
#   --output_model_path vit-t-cls-sample-block75.pt --batch_size 128 --max_device_batch_size 128
# python3 train_classifier.py --total_epoch 50 --pretrained_model_path vit-t-mae-sample-grid75-150ep.pt \
#   --output_model_path vit-t-cls-sample-grid75.pt --batch_size 128 --max_device_batch_size 128

# linear head

python3 train_classifier.py --total_epoch 50 --linear_probe --pretrained_model_path vit-t-mae-encmask-150ep.pt \
  --output_model_path vit-t-cls-lin-encmask.pt --run_tag vit-t-mae-encmask-150ep-lin \
  --batch_size 256 --max_device_batch_size 128

python3 train_classifier.py --total_epoch 50 --linear_probe --pretrained_model_path vit-t-mae-sample-random-150ep.pt \
  --output_model_path vit-t-cls-lin-random.pt --run_tag vit-t-mae-sample-random-150ep-lin \
  --batch_size 256 --max_device_batch_size 128
python3 train_classifier.py --total_epoch 50 --linear_probe --pretrained_model_path vit-t-mae-sample-block75-150ep.pt \
  --output_model_path vit-t-cls-lin-block75.pt --run_tag vit-t-mae-sample-block75-150ep-lin \
  --batch_size 256 --max_device_batch_size 128
python3 train_classifier.py --total_epoch 50 --linear_probe --pretrained_model_path vit-t-mae-sample-grid75-150ep.pt \
  --output_model_path vit-t-cls-lin-grid75.pt --run_tag vit-t-mae-sample-grid75-150ep-lin \
  --batch_size 256 --max_device_batch_size 128