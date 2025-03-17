API_NAME="gpt-4o"
MODE="combined" # select from ["timeseries_only", "combined"]
INDICATOR_TYPE="macd" # select from ["macd", "bb", "time"]
IN_DAYS=30
OUT_DAYS=7
# timeseries forecasting
python value_prediction.py \
    --dataset_folder="../../data/processed/finance/aligned_in${IN_DAYS}days_out${OUT_DAYS}days" \
    --save_path="../../results/finance/pred_${INDICATOR_TYPE}_in${IN_DAYS}_out${OUT_DAYS}/${API_NAME}/${MODE}" \
    --indicator=$INDICATOR_TYPE \
    --model=$API_NAME \
    --mode=$MODE