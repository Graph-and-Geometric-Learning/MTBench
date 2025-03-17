API_NAME="gpt-4o"
SETTING="long"

python correlation_prediction.py \
    --dataset_folder="../../data/processed/finance/QA${SETTING}" \
    --save_path="../../results/finance/correlation_${SETTING}/${API_NAME}" \
    --model=$API_NAME
    

python mcqa.py \
    --dataset_folder="../../data/processed/finance/QA${SETTING}" \
    --save_path="../../results/finance/mcqa_${SETTING}/${API_NAME}" \
    --model=$API_NAME

