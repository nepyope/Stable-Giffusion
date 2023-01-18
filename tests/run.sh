#!/bin/bash
export MODEL_NAME="flax/stable-diffusion-2-1"
export dataset_name="datatest"

python3 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="sd-test"

python3 generate_samples.py \
	--width 768 \
	--height 448 \
	--prompt "drone shot tracking around powerful geyser in rotorua" \
