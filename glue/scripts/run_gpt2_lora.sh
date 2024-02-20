#!/bin/bash

# Environment setup
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Experiment configuration
task=mnli
exp_name=gpt2_lora_$task
lr=5e-5
lr_ratio=20

# Execute command
python src/run_glue.py \
  --model_name_or_path gpt2 \
  --task_name $task \
  --use_lora \
  --loraplus_lr_ratio $lr_ratio \
  --target_modules "c_attn, c_proj, c_fc" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --max_eval_samples 128 \
  --eval_steps 2 \
  --save_steps 2 \
  --logging_steps 1 \
  --max_steps 3 \
  --learning_rate $lr \
  --optim adamw_torch \
  --lr_scheduler_type 'constant' \
  --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
  --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
  --evaluation_strategy steps \
  --save_strategy steps \
  --report_to tensorboard \
  --ignore_mismatched_sizes \
  --keep_checkpoints eval \
  --overwrite_output_dir \
  --save_total_limit 2
