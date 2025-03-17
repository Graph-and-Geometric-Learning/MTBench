import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.append("../..")

from meta_prompt import (
    temperature_forecast_metaprompt_generation,
    decode_temperature_forecast
)

from evaluation.api_call import (
    send_to_openai_chatgpt,
    send_to_openai_o1,
    send_to_deepseek,
    send_to_deepseek_r1,
    send_to_anthropic_claude,
    send_to_google_gemini,
)

from evaluation.utils import (
    save_to_json,
    plot_series
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, required=True, help="Dataset path")
parser.add_argument("--save_path", type=str, required=True, help="Results save path")
parser.add_argument("--model", type=str, default="gpt-4o", help="API model")
parser.add_argument("--mode", type=str, default="combined", help="choose from timeseries_only, text_only, combined")
parser.add_argument("--in_days", type=int, default=7, help="Input days")
parser.add_argument("--out_days", type=int, default=1, help="Output days")
args = parser.parse_args()

save_path = Path(args.save_path)
details_path = save_path / "output_details"
details_path.mkdir(parents=True, exist_ok=True)
visualizations_path = save_path / "visualizations"  
visualizations_path.mkdir(parents=True, exist_ok=True)
data_list = []
directory_path = Path(args.dataset_folder)
for json_file in directory_path.glob("*.json"):
    with open(json_file, "r") as file:
        data = json.load(file)
        extracted_data = {
            "filename": json_file.name,
            "input_window": data.get("input_window"),
            "output_window": data.get("output_window"),
            "text": data.get("text"),
            "input_timestamps": data.get("input_timestamps"),
        }
        data_list.append(extracted_data)

# data_list = data_list[:10]  # for testing

result_list = []
tot_samples = len(data_list)
print(f"Evaluating {tot_samples} samples...")

epoch_results = []
cumulative_mse, cumulative_mae, cumulative_rmse = [], [], []
for idx, sample in enumerate(data_list):
    try:
        filename = sample["filename"]
        output_ts = sample["output_window"]
        designed_prompt = temperature_forecast_metaprompt_generation(
            text=sample["text"],
            temperatures=sample["input_window"],
            start_datetime=sample["input_timestamps"][0],
            end_datetime=sample["input_timestamps"][-1],
            granularity="hourly",
            prediction_length=len(sample["output_window"]),
            mode=args.mode,
        )   

        if args.model == "gpt-4o":
            response = send_to_openai_chatgpt(designed_prompt, model="gpt-4o", max_tokens=None)
            answer = response.content.strip().replace('"', '')
        elif args.model == "sonnet":
            response = send_to_anthropic_claude(designed_prompt)
            answer = response.strip().replace('"', '')
        elif args.model == "gemini":
            response = send_to_google_gemini(designed_prompt,)
            answer = response.strip().replace('"', '')
        elif args.model == "deepseek":
            response = send_to_deepseek(designed_prompt)
            answer = response.strip().replace('"', '')
        elif args.model == "o1":
            response = send_to_openai_o1(designed_prompt)
            answer = response.content.strip().replace('"', '')
        elif args.model == "r1":
            response = send_to_deepseek_r1(designed_prompt)
            answer = response.content.strip().replace('"', '')
        else:
            raise NotImplementedError("This api is not supported")
        
            
        predict_ts = decode_temperature_forecast(answer)
        predict_ts_orig = predict_ts
        predict_ts = np.interp(
            np.linspace(0, 1, len(output_ts)), np.linspace(0, 1, len(predict_ts)), predict_ts
        )
        
        res = {
            "filename": sample["filename"],
            "response": answer,
            "ground_truth": output_ts,
            "predict": predict_ts.tolist(),
        }
        result_list.append(res)
        
        save_to_json(res, details_path / sample["filename"])

        first_half = sample["input_window"]
        plot_series(sample["filename"], first_half, output_ts, predict_ts_orig, visualizations_path)
        
        MAE = mean_absolute_error(output_ts, predict_ts)
        MSE = mean_squared_error(output_ts, predict_ts)
        RMSE = np.sqrt(MSE)
        
        if MSE > 100:
            epoch_results.append({
                "filename": sample["filename"],
                "failed": True,
                "epoch": idx + 1,
                "mse": MSE,
                "mae": MAE,
                "rmse": RMSE
            })
            continue
        
        cumulative_mse.append(MSE)
        cumulative_mae.append(MAE)
        cumulative_rmse.append(RMSE)
        epoch_results.append({
            "filename": sample["filename"],
            "epoch": idx + 1,
            "mse": MSE,
            "mae": MAE,
            "rmse": RMSE,
            "mean_mse": np.mean(cumulative_mse),
            "mean_mae": np.mean(cumulative_mae),
            "mean_rmse": np.mean(cumulative_rmse),
        })
        save_to_json(epoch_results, f"{save_path}/epoch_results.json")
        print("{}/{}: mse: {:.4f}, mae: {:.4f}, rmse: {:.4f}".format(idx, tot_samples, MSE, MAE, RMSE))

    except Exception as e:
        print(f"Skipping {idx} due to error: {e}")


# Compute final statistics
summary = {
    "total_samples": len(result_list),
    "mse": np.mean(cumulative_mse),
    "mae": np.mean(cumulative_mae),
    "rmse": np.mean(cumulative_rmse)
}

save_to_json(summary, f"{save_path}/final_results.json")
print(f"Processing complete. Results saved to {save_path}/final_results.json")
