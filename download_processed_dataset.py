from huggingface_hub import snapshot_download

# downloading processed finance dataset

## For Time Series Forecasting,Trend Prediction and Technical Indicator
snapshot_download("afeng/MTBench_finance_aligned_pairs_short", local_dir="./data/processed/finance/aligned_in7days_out1days", repo_type="dataset")
snapshot_download("afeng/MTBench_finance_aligned_pairs_long", local_dir="./data/processed/finance/aligned_in30days_out7days", repo_type="dataset")

## For MCQA and Correlation
snapshot_download("afeng/MTBench_finance_QA_short", local_dir="./data/processed/finance/QAshort", repo_type="dataset")
snapshot_download("afeng/MTBench_finance_QA_long", local_dir="./data/processed/finance/QAlong", repo_type="dataset")



# downloading processed weather dataset

## For Time Series Forecasting,Trend Prediction and Technical Indicator
snapshot_download("afeng/MTBench_weather_aligned_pairs_short", local_dir="./data/processed/weather/aligned_in7days_out1days", repo_type="dataset")
snapshot_download("afeng/MTBench_weather_aligned_pairs_long", local_dir="./data/processed/weather/aligned_in14days_out3days", repo_type="dataset")

## For MCQA
snapshot_download("afeng/MTBench_weather_QA_short", local_dir="./data/processed/weather/QAshort", repo_type="dataset")
snapshot_download("afeng/MTBench_weather_QA_long", local_dir="./data/processed/weather/QAlong", repo_type="dataset")