import sys
from pathlib import Path
import numpy as np
import random
import argparse
import os
from datetime import datetime

sys.path.append("../..")

from utils import (
    plot_timeseries_with_reference,
    sample_a_stock_pair_flexible_units_given_event,
    calculate_stock_trend,
    save_dict_to_local_json,
    read_folder_of_json,
    convert_iso8601_to_datetime_and_timestamp,
    read_json_from_local,
)

from utils import (
    compute_sma,
    compute_ema,
    compute_macd,
    compute_bollinger_bands,
)

def return_news_and_ts_sentiment(news_label_list, ts_bin_label):
    news_sentiment_mapping = {
        "[1][a] Bullish": "positive",
        "[1][b] Growth-Oriented": "positive",
        "[1][c] Upbeat Market Reaction": "positive",
        "[2][a] Balanced/Informational": "neutral",
        "[2][b] Mixed Outlook": "neutral",
        "[2][c] Speculative": "neutral",
        "[3][a] Bearish": "negative",
        "[3][b] Risk & Warning": "negative",
        "[3][c] Market Panic/Fear": "negative",
    }

    timeseries_sentiment_mapping = {
        "<-4%": "negative",
        "-2% ~ -4%": "negative",
        "-2% ~ +2%": "neutral",
        "+2% ~ +4%": "positive",
        ">+4%": "positive",
    }
    return {"news_sentiment_list": [news_sentiment_mapping[n] for n in news_label_list], "ts_sentiment": timeseries_sentiment_mapping[ts_bin_label]}


def calculate_technical_indicators(ts):
    # Compute moving averages and indicators for input and output windows
    sma_10 = compute_sma(ts, 10)
    sma_50 = compute_sma(ts, 50)
    sma_200 = compute_sma(ts, 200)

    ema_10 = compute_ema(ts, 10)
    ema_50 = compute_ema(ts, 50)
    ema_200 = compute_ema(ts, 200)

    macd, signal = compute_macd(ts)
    upper_bb, lower_bb = compute_bollinger_bands(ts)

    return {
        "sma_10": sma_10,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "ema_10": ema_10,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "macd": macd,
        "signal": signal,
        "upper_bb": upper_bb,
        "lower_bb": lower_bb,
    }


# Argument Parsing
parser = argparse.ArgumentParser(description="Timeseries Stock Analysis and Prediction")

parser.add_argument(
    "--ts_folder",
    type=str,
    default="/home/ubuntu/time/data/timeseries/stock_ts_aligned_10years",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="/home/ubuntu/time/data/task_curation/ts_value_pred/out",
)
parser.add_argument("--input_window_size", type=int, default=1)
parser.add_argument("--input_window_unit", type=str, default="days")
parser.add_argument("--input_granularity", type=int, default=1)
parser.add_argument("--input_granularity_unit", type=str, default="hours")
parser.add_argument("--output_window_size", type=int, default=1)
parser.add_argument("--output_window_unit", type=str, default="days")
parser.add_argument("--output_granularity", type=int, default=1)
parser.add_argument("--output_granularity_unit", type=str, default="hours")
parser.add_argument("--text_folder", type=str, required=True)

# Parse arguments
args = parser.parse_args()

# Ensure output directories exist
Path(args.save_folder).mkdir(parents=True, exist_ok=True)
Path(args.save_folder, "visualization").mkdir(parents=True, exist_ok=True)

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define bins and labels
bins = [-np.inf, -4, -2, 2, 4, np.inf]
labels = ["<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%"]

# Load news articles
news_list = read_folder_of_json(args.text_folder)
if not news_list:
    print("No news articles found in the provided text path.")
    sys.exit(1)

# Collect all available stock tickers
all_ticker_set = {
    file.stem for file in Path(args.ts_folder).glob("*.json") if file.is_file()
}
if not all_ticker_set:
    print("Error: No stock time-series data found in the specified folder.")
    sys.exit(1)

# Shuffle news list for random sampling
random.shuffle(news_list)

# Processing Loop
global_idx = 0
success_cnt = 0

for news in news_list:
    if "content" not in news or "tickers" not in news:
        continue

    related_ticker_list = news["tickers"]
    if not all(ticker in all_ticker_set for ticker in related_ticker_list):
        print(f"Skipping news {related_ticker_list}: Related tickers not found in time-series data.")
        continue

    for ticker_name in related_ticker_list:
        target_ts_file = Path(args.ts_folder) / f"{ticker_name}.json"

        try:
            ts = read_json_from_local(target_ts_file)
            dt, timestamp_ms = convert_iso8601_to_datetime_and_timestamp(
                news["published_utc"]
            )

            if not (ts["real_timestamp"][0] <= timestamp_ms <= ts["real_timestamp"][-1]):
                print(f"Skipping {ticker_name}: News published outside time-series range.")
                continue

            ############## Sampling ts pair based on news time ##############
            pair = sample_a_stock_pair_flexible_units_given_event(
                ts["real_timestamp"],
                ts["open"],
                timestamp_ms,
                event_position_ratio=0.9,
                input_window_size=args.input_window_size,
                input_window_unit=args.input_window_unit,
                input_granularity=args.input_granularity,
                input_granularity_unit=args.input_granularity_unit,
                output_window_size=args.output_window_size,
                output_window_unit=args.output_window_unit,
                output_granularity=args.output_granularity,
                output_granularity_unit=args.output_granularity_unit,
            )


            ############### get the associated news info ##############
            text_dict = {
                "content": news["content"],
                "timestamp_ms": timestamp_ms,
                "published_utc": news["published_utc"],
                "article_url": news.get("article_url", "N/A"),
                "label_type": news.get("label_type", "N/A"),
                "label_time": news.get("label_time", "N/A"),
                "label_sentiment": news.get("label_sentiment", "N/A"),
            }
            pair.update({"text": text_dict})

            ################ Calculate stock trend ################
            res_trend = calculate_stock_trend(
                pair["input_timestamps"],
                pair["input_window"],
                pair["output_timestamps"],
                pair["output_window"],
                statistic="mean",
                bins=bins,
                labels=labels,
            )

            trend_dict = {
                "input_percentage_change": res_trend["input_percentage_change"],
                "input_bin_label": res_trend["input_bin_label"],
                "output_percentage_change": res_trend["output_percentage_change"],
                "output_bin_label": res_trend["output_bin_label"],
                "overall_percentage_change": res_trend["overall_percentage_change"],
                "overall_bin_label": res_trend["overall_bin_label"],
            }
            pair.update({"trend": trend_dict})


            ################ Calculate technical indicators ################
            technical_in = calculate_technical_indicators(pair["input_window"])
            technical_dict = {}
            technical_dict.update(
                {
                    "in_sma_10": technical_in["sma_10"],
                    "in_sma_50": technical_in["sma_50"],
                    "in_sma_200": technical_in["sma_200"],
                    "in_ema_10": technical_in["ema_10"],
                    "in_ema_50": technical_in["ema_50"],
                    "in_ema_200": technical_in["ema_200"],
                    "in_macd": technical_in["macd"],
                    "in_signal": technical_in["signal"],
                    "in_upper_bb": technical_in["upper_bb"],
                    "in_lower_bb": technical_in["lower_bb"],
                }
            )

            technical_out = calculate_technical_indicators(pair["output_window"])

            technical_dict.update(
                {
                    "out_sma_10": technical_out["sma_10"],
                    "out_sma_50": technical_out["sma_50"],
                    "out_sma_200": technical_out["sma_200"],
                    "out_ema_10": technical_out["ema_10"],
                    "out_ema_50": technical_out["ema_50"],
                    "out_ema_200": technical_out["ema_200"],
                    "out_macd": technical_out["macd"],
                    "out_signal": technical_out["signal"],
                    "out_upper_bb": technical_out["upper_bb"],
                    "out_lower_bb": technical_out["lower_bb"],
                }
            )

            technical_overall = calculate_technical_indicators(
                pair["input_window"] + pair["output_window"]
            )

            technical_dict.update(
                {
                    "overall_sma_10": technical_overall["sma_10"],
                    "overall_sma_50": technical_overall["sma_50"],
                    "overall_sma_200": technical_overall["sma_200"],
                    "overall_ema_10": technical_overall["ema_10"],
                    "overall_ema_50": technical_overall["ema_50"],
                    "overall_ema_200": technical_overall["ema_200"],
                    "overall_macd": technical_overall["macd"],
                    "overall_signal": technical_overall["signal"],
                    "overall_upper_bb": technical_overall["upper_bb"],
                    "overall_lower_bb": technical_overall["lower_bb"],
                }
            )
            pair.update({"technical": technical_dict})
            
            
            
            ###

            sentiments = return_news_and_ts_sentiment(news["label_sentiment"], res_trend["output_bin_label"])
            
            if sentiments["ts_sentiment"] in sentiments["news_sentiment_list"]:
                pair["alignment"] = "consistent"
            else:
                pair["alignment"] = "inconsistent"

            
            ########### Print processed information ###########
            
            print(
                f"✔ Processed {success_cnt}: {ticker_name}, Trend: {res_trend['output_percentage_change']}, Label: {res_trend['output_bin_label']}, {pair['alignment']}"
            )

            # Save results
            save_name = f"{global_idx}_{ticker_name}"
            save_dict_to_local_json(pair, Path(args.save_folder) / f"{save_name}.json")

            # Generate visualization
            plot_timeseries_with_reference(
                [
                    (
                        [datetime.fromtimestamp(x) for x in pair["input_timestamps"]],
                        pair["input_window"],
                        [datetime.fromtimestamp(x) for x in pair["output_timestamps"]],
                        pair["output_window"],
                    )
                ],
                save_path=Path(args.save_folder, "visualization", f"{save_name}.png"),
                reference_timeseries=(
                    [datetime.fromtimestamp(x / 1000.0) for x in ts["real_timestamp"]],
                    ts["open"],
                ),
                event_points=[datetime.fromtimestamp(timestamp_ms / 1000.0)],
                event_descriptions=[f"News URL: {pair['text']['article_url']}"],
                stretch=10,
            )

            success_cnt += 1
            global_idx += 1

        except Exception as e:
            print(f"⚠ Error processing {ticker_name}: {e}")
            global_idx += 1
            continue
