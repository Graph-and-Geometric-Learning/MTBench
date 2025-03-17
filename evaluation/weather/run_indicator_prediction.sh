API_NAME="gpt-4o"
MODE="combined" # select from ["timeseries_only", "combined"]

IN_DAYS=7
OUT_DAYS=1

python indicator_prediction.py \
    --dataset_folder="../../data/processed/weather/aligned_in${IN_DAYS}days_out${OUT_DAYS}days" \
    --save_path="../../results/weather/indicator_prediction_in${IN_DAYS}_out${OUT_DAYS}/${API_NAME}/${MODE}" \
    --model=$API_NAME \
    --mode=$MODE \
    --in_days=$IN_DAYS \
    --out_days=$OUT_DAYS
