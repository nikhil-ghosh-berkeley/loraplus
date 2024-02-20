#!/bin/bash

# Environment setup
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Experiment configuration
task=mnli
exp_name=llama-7b_lora_$task
lr=1e-4
lr_ratio=1

# Execute command
python src/run_glue.py \
  --model_name_or_path huggyllama/llama-7b \
  --task_name $task \
  --use_lora \
  --target_modules "q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --max_seq_length 128 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --max_eval_samples 128 \
  --eval_steps 2 \
  --save_steps 2 \
  --logging_steps 1 \
  --max_steps 3 \
  --learning_rate $lr \
  --loraplus_lr_ratio $lr_ratio \
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
