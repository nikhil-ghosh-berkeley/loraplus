#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

exp_name=gpt2_lora_debug
lr=5e-5
lr_ratio=20

python ~/loraplus/src/run_glue.py \
    --model_name_or_path gpt2 \
    --use_lora \
    --loraplus_lr_ratio $lr_ratio \
    --target_modules "c_attn, c_proj, c_fc" \
    --task_name mnli \
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
    --save_total_limit 2 \
    --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
    --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
    --evaluation_strategy steps \
    --save_strategy steps \
    --seed 0 \
    --report_to none \
    --ignore_mismatched_sizes \
    --keep_checkpoints eval \
    --overwrite_output_dir