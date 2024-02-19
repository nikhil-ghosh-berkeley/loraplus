#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

exp_name=roberta_lora_debug
lr=5e-5
lr_ratio=20

python src/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mnli \
    --use_lora \
    --target_modules "query, key" \
    --do_train \
    --save_total_limit 4 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --max_eval_samples 128 \
    --eval_steps 2 \
    --save_steps 2 \
    --logging_steps 1 \
    --max_steps 3 \
    --learning_rate $lr \
    --loraplus_lr_ratio $lr_ratio \
    --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
    --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
    --evaluation_strategy steps \
    --save_strategy steps \
    --warmup_ratio 0.06 \
    --gradient_accumulation_steps 1 \
    --seed 1 \
    --report_to none \
    --keep_checkpoints all \
    --overwrite_output_dir \
    --ignore_mismatched_sizes