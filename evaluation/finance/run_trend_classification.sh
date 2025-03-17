API_NAME="gpt-4o"
MODE="combined" # select from ["timeseries_only", "combined"]
IN_DAYS=30
OUT_DAYS=7

python trend_classification.py \
    --dataset_folder="../../data/processed/finance/aligned_in${IN_DAYS}days_out${OUT_DAYS}days" \
    --save_path="../../results/finance/trend_classification_in${IN_DAYS}_out${OUT_DAYS}/${API_NAME}/${MODE}" \
    --model=$API_NAME \
    --mode=$MODE
