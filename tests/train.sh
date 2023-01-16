#!/bin/bash
export MODEL_NAME="flax/stable-diffusion-2-1"
export dataset_name="datatest"

python3 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=50000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="sd-test"
