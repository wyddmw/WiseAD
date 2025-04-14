#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH=${WORK_DIR}

ARCH=$1
TASK=$2
case ${TASK} in
    "pretrain-finetune-test")
        cd ${WORK_DIR}
        OUTPUT_DIR=${WORK_DIR}/outputs/${ARCH}_$(date +"%Y%m%d_%H%M%S")
        mkdir -p ${OUTPUT_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        bash run.sh ${ARCH} pretrain ${LANGUAGE_MODEL} ${VISION_MODEL} ${OUTPUT_DIR}
        bash run.sh ${ARCH} finetune ${LANGUAGE_MODEL} ${VISION_MODEL} ${OUTPUT_DIR}
        bash run.sh ${ARCH} test ${OUTPUT_DIR}/mobilevlm_v2-2.finetune
    ;;
    "pretrain")
        echo ">>> Start Pre-training ..."
        cd ${WORK_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        OUTPUT_DIR=$5
        OUTPUT_DIR_PT=${OUTPUT_DIR}/mobilevlm_v2-1.pretrain
        mkdir -p ${OUTPUT_DIR_PT}
        deepspeed mobilevlm/train/train_mem.py \
            --deepspeed scripts/deepspeed/zero2.json \
            --model_name_or_path ${LANGUAGE_MODEL} \
            --version plain \
            --data_path data/pretrain_data/share-captioner_coco_lcs_sam_1246k_1107.json \
            --image_folder data/pretrain_data \
            --vision_tower ${VISION_MODEL} \
            --vision_tower_type clip \
            --mm_projector_type ldpnetv2 \
            --mm_projector_lr 1e-3 \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --bf16 True \
            --output_dir ${OUTPUT_DIR_PT} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 24000 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to none \
            2>&1 | tee ${OUTPUT_DIR_PT}/log.txt &&
        echo "Done."
    ;;
    "finetune")
        echo ">>> Start Multi-task Training ..."
        cd ${WORK_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        OUTPUT_DIR=$5
        OUTPUT_DIR_PT=${OUTPUT_DIR}/mobilevlm_v2-1.pretrain
        OUTPUT_DIR_FT=${OUTPUT_DIR}/mobilevlm_v2-2.finetune
        mkdir -p ${OUTPUT_DIR_FT}
        deepspeed --include localhost:0,1,2,3 mobilevlm/train/train_mem.py \
            --deepspeed scripts/deepspeed/zero3.json \
            --model_name_or_path ${LANGUAGE_MODEL} \
            --version v1 \
            --data_path data/carla/carla_qa.json \
            --image_folder ./ \
            --vision_tower ${VISION_MODEL} \
            --vision_tower_type clip \
            --mm_projector_type ldpnetv2 \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length False \
            --bf16 True \
            --output_dir ${OUTPUT_DIR_FT} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 2000 \
            --save_total_limit 1 \
            --learning_rate 4e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb \
            --dataset_name joint_LingoQA_Carla_DRAMA \
            --drama_data_path data/DRAMA/DRAMA_qa.json \
            --drama_data_repeat 1 \
            --lingoqa_data_path data/LingoQA/training_data.json \
            2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
        echo "Done."
    ;;
    "finetune_continue")
        echo ">>> Start Multi-task Training ..."
        cd ${WORK_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        OUTPUT_DIR=$5
        DRAMA_JSON=$6
        OUTPUT_DIR_PT=${OUTPUT_DIR}/mobilevlm_v2-1.pretrain
        OUTPUT_DIR_FT=${OUTPUT_DIR}/mobilevlm_v2-2.finetune
        mkdir -p ${OUTPUT_DIR_FT}
        deepspeed --include localhost:0,1,2,3 mobilevlm/train/train_mem.py \
            --deepspeed scripts/deepspeed/zero3.json \
            --model_name_or_path ${LANGUAGE_MODEL} \
            --version v1 \
            --data_path data/carla/carla_qa.json \
            --image_folder ./ \
            --vision_tower ${VISION_MODEL} \
            --vision_tower_type clip \
            --mm_projector_type ldpnetv2 \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length False \
            --bf16 True \
            --output_dir ${OUTPUT_DIR_FT} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 2000 \
            --save_total_limit 1 \
            --learning_rate 1e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.1 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb \
            --dataset_name joint_LingoQA_Carla_DRAMA \
            --drama_data_path data/DRAMA/DRAMA_qa.json \
            --drama_data_repeat 1 \
            --lingoqa_data_path data/LingoQA/training_data.json \
            2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
        echo "Done."
    ;;
    "test")
        echo ">>> Start Evaluation ..."
        cd ${WORK_DIR}
        OUTPUT_DIR=$3
        bash scripts/benchmark.sh ${OUTPUT_DIR}
    ;;    
    "finetune.lora")
        echo ">>> Start Visual-Instruction Tuning with LoRA..."
        cd ${WORK_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        OUTPUT_DIR=$5
        OUTPUT_DIR_PT=${OUTPUT_DIR}/mobilevlm_v2-1.pretrain
        OUTPUT_DIR_FT=${OUTPUT_DIR}/mobilevlm_v2-2.finetune-lora
        mkdir -p ${OUTPUT_DIR_FT}
        declare -A DS_CONF
        deepspeed --master_port 29502 --include localhost:0,1,2,3 mobilevlm/train/train_mem.py \
            --model_name_or_path ${LANGUAGE_MODEL} \
            --deepspeed scripts/deepspeed/zero3.json \
            --lora_enable True --lora_r 128 --lora_alpha 256 \
            --learning_rate 4e-5 \
            --version v1 \
            --data_path data/carla/carla_qa.json \
            --image_folder ./ \
            --vision_tower ${VISION_MODEL} \
            --vision_tower_type clip \
            --mm_projector_type ldpnetv2 \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length False \
            --bf16 True \
            --output_dir ${OUTPUT_DIR_FT} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50000 \
            --save_total_limit 1 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --dataset_name joint_LingoQA_Carla_DRAMA \
            --drama_data_path data/DRAMA/DRAMA_qa.json \
            --drama_data_repeat 1 \
            --lingoqa_data_path data/LingoQA/training_data.json \
            --report_to wandb \
            2>&1 | tee -a ${LANGUAGE_MODEL}/log.txt &&
        python3 scripts/mergelora.py ${LANGUAGE_MODEL} ${OUTPUT_DIR}/mobilevlm_v2-2.finetune-lora ${OUTPUT_DIR}/mobilevlm_v2-2.finetune \
            2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
        echo "Done."
    ;;
    "finetune.lora_finetune")
        echo ">>> Start Visual-Instruction Tuning with LoRA..."
        cd ${WORK_DIR}
        LANGUAGE_MODEL=$3
        VISION_MODEL=$4
        OUTPUT_DIR=$5
        OUTPUT_DIR_PT=${OUTPUT_DIR}/mobilevlm_v2-1.pretrain
        OUTPUT_DIR_FT=${OUTPUT_DIR}/mobilevlm_v2-2.finetune-lora
        mkdir -p ${OUTPUT_DIR_FT}
        declare -A DS_CONF
        deepspeed --master_port 29501 --include localhost:0,1,2,3 mobilevlm/train/train_mem.py \
            --model_name_or_path ${LANGUAGE_MODEL} \
            --deepspeed scripts/deepspeed/zero3.json \
            --lora_enable True --lora_r 128 --lora_alpha 256 \
            --learning_rate 1e-5 \
            --version v1 \
            --data_path data/carla/carla_qa.json \
            --image_folder ./ \
            --vision_tower ${VISION_MODEL} \
            --vision_tower_type clip \
            --mm_projector_type ldpnetv2 \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length False \
            --bf16 True \
            --output_dir ${OUTPUT_DIR_FT} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50000 \
            --save_total_limit 1 \
            --weight_decay 0. \
            --warmup_ratio 0.1 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --dataset_name joint_LingoQA_Carla_DRAMA \
            --drama_data_path data/DRAMA/DRAMA_qa.json \
            --drama_data_repeat 1 \
            --lingoqa_data_path data/LingoQA/training_data.json \
            --report_to wandb \
            2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
        python3 scripts/mergelora.py ${LANGUAGE_MODEL} ${OUTPUT_DIR}/mobilevlm_v2-2.finetune-lora ${OUTPUT_DIR}/mobilevlm_v2-2.finetune \
            2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
        echo "Done."
    ;;
    *)
        echo "error with ${DATASET_ID}"
esac
