#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

exp_name=llama_lora_debug
lr=1e-4
lr_ratio=1

python src/run_glue.py \
    --model_name_or_path huggyllama/llama-7b \
    --use_lora \
    --target_modules "q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj" \
    --task_name mnli \
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
    --save_total_limit 2 \
    --output_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio} \
    --logging_dir output/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
    --evaluation_strategy steps \
    --save_strategy steps \
    --seed 1 \
    --report_to none \
    --ignore_mismatched_sizes \
    --keep_checkpoints eval \
    --overwrite_output_dir