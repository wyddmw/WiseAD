# Modify the following paths to your own paths
VISION_MODEL=/your/path/to/clip-vit-large-patch14-336
LANGUAGE_MODEL=/your/path/to/MobileVLM_V2-1.7B
bash train.sh mobilevlm_v2_1.7b finetune ${LANGUAGE_MODEL} ${VISION_MODEL} ./WiseAD/epoch1

# # # Continue training for additional epochs
LANGUAGE_MODEL=WiseAD/epoch1/mobilevlm_v2-2.finetune
bash train.sh mobilevlm_v2_1.7b finetune_continue ${LANGUAGE_MODEL} ${VISION_MODEL} ./WiseAD/epoch2