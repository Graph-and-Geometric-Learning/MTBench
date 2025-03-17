from huggingface_hub import snapshot_download

# downloading raw finance dataset
snapshot_download("afeng/MTBench_finance_news", local_dir="./data/raw/finance/news_labeled_20000", repo_type="dataset")
snapshot_download("afeng/MTBench_finance_stock", local_dir="./data/raw/finance/stock_ts", repo_type="dataset")


# downloading raw weather dataset
snapshot_download("afeng/MTBench_weather_news", local_dir="./data/raw/weather/text", repo_type="dataset")
snapshot_download("afeng/MTBench_weather_temperature", local_dir="./data/raw/weather/timeseries", repo_type="dataset")

