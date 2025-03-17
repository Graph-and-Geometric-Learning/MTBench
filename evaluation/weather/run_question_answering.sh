API_NAME="gpt-4o"
SETTING="long" # select from ["long", "short"]

python mcqa.py \
    --dataset_folder="../../data/processed/weather/QA${SETTING}" \
    --save_path="../../results/weather/question_answering_${SETTING}/${API_NAME}" \
    --model=$API_NAME \
    --setting=$SETTING
