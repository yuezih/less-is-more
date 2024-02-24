################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

RUN_NAME=llava-$MODEL_VERSION-finetune_qlora-150k_filtered

# NOTE: Since the filtered data is not as sufficient as the original data, we suggest keep the same training step number as the original finetuning process, instead of using the same number of epochs.

# CUDA_VISIBLE_DEVICES=4 \
WANDB_PROJECT=LLaVA \
deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 29500 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --bits 4 \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/llava_instruct_150k_filtered.json \
    --image_folder /path/to/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-pretrain-$MODEL_VERSION/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoint/$RUN_NAME \
    --max_steps 3696 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name $RUN_NAME