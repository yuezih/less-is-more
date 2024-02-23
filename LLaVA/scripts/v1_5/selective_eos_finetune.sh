#!/bin/bash

MODEL_NAME=llava-v1.5-7b-task-lora-detail23k

WANDB_PROJECT=LLaVA \
deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 29500 \
    llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-6 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path llava-v1.5-7b \
    --output_dir ./checkpoint/$MODEL_NAME \
    --image_folder ./playground/data/coco/train2017 \
    --data_path ./playground/LLaVA-Instruct-150K/detail_23k.json \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --run_name $MODEL_NAME \
    --version v1 \
    --vision_tower clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 200 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True