WISEAD_MODEL=$1
OUTPUT_PATH=$2
# generate predictions
python eval/LingoQA/run_inference.py --model_path ${WISEAD_MODEL} --output_path ${OUTPUT_PATH}
# generate .csv file for evaluation
python eval/LingoQA/generate_csv.py --input_folder_path ${OUTPUT_PATH}/ --output_folder_path ${OUTPUT_PATH}
# run evaluation with LingoQA-Judge metric
python eval/LingoQA/eval_metric.py --predictions_path ${OUTPUT_PATH}/lingoqa_results.csv