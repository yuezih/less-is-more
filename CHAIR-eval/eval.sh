RUN_NAME=your-model-folder
MODEL_NAME=LLaVA/checkpoint/$RUN_NAME
MODEL_BASE=llava-v1.5-7b

echo "========================="
echo "MODEL: $RUN_NAME"
echo "========================="

CUDA_VISIBLE_DEVICES=0 \
python LLaVA/llava/eval/model_vqa.py \
--model-path $MODEL_NAME \
--model-base $MODEL_BASE \
--question-file data/chair-500.jsonl \
--image-folder data/chair-500 \
--answers-file $MODEL_NAME/answer-chair.jsonl \
--temperature 0

python chair.py \
--coco_path CHAIR-eval/MSCOCO/annotations \
--cache CHAIR-eval/data/chair.pkl \
--cap_file $MODEL_NAME/answer-chair.jsonl \
--save_path $MODEL_NAME/eval-chair.json

echo "========================="
echo "MODEL: $RUN_NAME"
echo "========================="