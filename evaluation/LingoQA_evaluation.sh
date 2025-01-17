export CUDA_VISIBLE_DEVICES=0
OUTPUT_FILE=$2
MODEL_DIR=$1

python scripts/eval_LingoQA.py --eval_file /data2/LLM-Datasets/LingoQA/evaluation_data.json \
    --model-path ${MODEL_DIR} \
    --output_file ${OUTPUT_FILE}