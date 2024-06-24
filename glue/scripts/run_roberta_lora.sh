#!/bin/bash

# Environment setup
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1

# Experiment configuration
task=mnli
exp_name=roberta_lora_$task
lr=5e-5
lr_ratio=80

# Execute command
python src/run_glue.py \
  --model_name_or_path roberta-base \
  --task_name $task \
  --use_lora \
  --target_modules "query, value" \
  --do_train \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --max_seq_length 128 \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 10 \
  --num_train_epochs 3 \
  --learning_rate $lr \
  --loraplus_lr_ratio $lr_ratio \
  --lora_rank 8 \
  --lora_alpha 8 \
  --fp16 \
  --lr_scheduler_type 'linear' \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --adam_epsilon 1e-8 \
  --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
  --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
  --evaluation_strategy steps \
  --save_strategy steps \
  --report_to tensorboard \
  --keep_checkpoints eval \
  --overwrite_output_dir \
  --ignore_mismatched_sizes \
  --seed 1 \
  --lora_use_original_init \
  --save_total_limit 1
