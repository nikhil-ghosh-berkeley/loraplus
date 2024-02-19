
export WANDB_PROJECT=llama-7b_lora_mnli_sweep_test
export TOKENIZERS_PARALLELISM=false

lr=${1:-0.0001}
lr_ratio=${2:-1}
seed=${3:-0}
export CUDA_VISIBLE_DEVICES=${4:-0}

export WANDB_RUN_GROUP=lr_${lr}_ratio_${lr_ratio}
export WANDB_NAME=${WANDB_RUN_GROUP}_seed_${seed}

python src/run_glue.py \
    --model_name_or_path huggyllama/llama-7b \
    --task_name mnli \
    --output_dir ./output/$WANDB_PROJECT/$WANDB_NAME \
    --use_lora \
    --target_modules "q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj" \
    --lr_scheduler_type constant \
    --optim adamw_torch \
    --learning_rate $lr \
    --loraplus_lr_ratio $lr_ratio \
    --adam_beta2 0.999 \
    --weight_decay 0.0 \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --logging_steps 10 \
    --logging_strategy steps \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing \
    --fp16 \
    --report_to tensorboard wandb \
    --keep_checkpoints eval \
    --ignore_mismatched_sizes \
    --seed $seed