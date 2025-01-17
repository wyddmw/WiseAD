# # CODA evaluation
# export CUDA_VISIBLE_DEVICES=0
# MODEL_PATH=$1
# output_path=$2
# python scripts/eval_LingoQA.py --eval_file data/CODA/CODA-LM/Train/vqa_anno/CODA_eval.json \
#     --model-path ${MODEL_PATH} \
#     --output_path ${output_path}

# BDDX driving analysis evaluation
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=$1
MODEL=$2
OUTPUT_PATH=/data2/MobileVLM_Drive_models/comprehensive_comparions/${MODEL}
python scripts/eval_LingoQA.py --eval_file data/BDDX/bddx-action.json \
    --model-path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH}

# BDDX driving action evaluation
python scripts/eval_LingoQA.py --eval_file data/BDDX/bddx-ana.json \
    --model-path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH}

# # DriveLM driving suggestion evaluation
# python scripts/eval_LingoQA.py --eval_file data/BDDX/conversation_bddx_sampled.json \
#     --model-path ${MODEL_PATH} \
#     --output_path ${output_path}

# # DriveLM object recognition evaluation
# export CUDA_VISIBLE_DEVICES=0
# python scripts/eval_LingoQA.py --eval_file data/BDDX/conversation_bddx_sampled.json \
#     --model-path ${MODEL_PATH} \
#     --output_path ${output_path}