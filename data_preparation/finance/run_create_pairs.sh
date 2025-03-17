## 7days -> 1day
INPUT_WINDOW_SIZE=7
INPUT_WINDOW_UNIT="days"
INPUT_GRANULARITY=5
INPUT_GRANULARITY_UNIT="minutes"

OUTPUT_WINDOW_SIZE=1
OUTPUT_WINDOW_UNIT="days"
OUTPUT_GRANULARITY=5
OUTPUT_GRANULARITY_UNIT="minutes"


IN_TS_FOLDER="../../data/raw/finance/stock_ts"
IN_TEXT_FOLDER="../../data/raw/finance/news_labeled_20000"
SAVE_FOLDER="../../data/processed/finance/pair_in_${INPUT_WINDOW_SIZE}${INPUT_WINDOW_UNIT}_${INPUT_GRANULARITY}${INPUT_GRANULARITY_UNIT}_out_${OUTPUT_WINDOW_SIZE}${OUTPUT_WINDOW_UNIT}_${OUTPUT_GRANULARITY}${OUTPUT_GRANULARITY_UNIT}"

python create_pairs.py  \
    --text_folder=$IN_TEXT_FOLDER \
    --ts_folder=$IN_TS_FOLDER \
    --save_folder=$SAVE_FOLDER  \
    --input_window_size=$INPUT_WINDOW_SIZE \
    --input_window_unit=$INPUT_WINDOW_UNIT \
    --input_granularity=$INPUT_GRANULARITY \
    --input_granularity_unit=$INPUT_GRANULARITY_UNIT \
    --output_window_size=$OUTPUT_WINDOW_SIZE \
    --output_window_unit=$OUTPUT_WINDOW_UNIT \
    --output_granularity=$OUTPUT_GRANULARITY \
    --output_granularity_unit=$OUTPUT_GRANULARITY_UNIT 