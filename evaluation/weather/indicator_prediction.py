import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append("../..")
import os
from meta_prompt import  temperature_indicator_metaprompt_generation, decode_temperature_indicator
from evaluation.utils import (
    save_to_json, get_temperature_diff_max_min
)
from evaluation.api_call import (
    send_to_openai_chatgpt,
    send_to_openai_o1,
    send_to_deepseek,
    send_to_deepseek_r1,
    send_to_anthropic_claude,
    send_to_google_gemini,
    send_to_llama
)
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_folder", type=str, default="", help=""
)
parser.add_argument(
    "--save_path", type=str, default="out-sampled", help=""
)

parser.add_argument(
    "--model",  type=str, default="gpt-4o", help=""
)
parser.add_argument(
    "--mode",
    type=str,
    default="combined",
    help="choose from timeseries_only, text_only, combined",
)
parser.add_argument("--in_days", type=int, default=7, help="Input days")
parser.add_argument("--out_days", type=int, default=1, help="Output days")

args = parser.parse_args()

directory_path = Path(args.dataset_folder)

data_list = []

if args.model == "llama":
    import transformers
    import torch
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

# Loop through all JSON files in the directory
for json_file in directory_path.glob("*.json"):
    with open(json_file, 'r') as file:
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

os.makedirs(Path(args.save_path).parent, exist_ok=True)
result_list = []
tot_samples = len(data_list)
print("Evaluating {} samples......".format(tot_samples))
metric_dict = {"diff": {"mse": [], "mae": []},
               "max": {"mse": [], "mae": []},
               "min": {"mse": [], "mae": []}}
for idx, sample in enumerate(data_list):
    filename = sample["filename"]
    output_ts = sample["output_window"]
    
    designed_prompt = temperature_indicator_metaprompt_generation(
        text=sample["text"],
        past_temperatures=sample["input_window"],
        start_datetime=sample["input_timestamps"][0],
        end_datetime=sample["input_timestamps"][-1],
        granularity="hourly",
        past_days=args.in_days,
        mode=args.mode)
    try:
        if args.model == "gpt-4o":
            response = send_to_openai_chatgpt(designed_prompt, model="gpt-4o", max_tokens=None)
            answer = response.content.strip().replace('"', '')
        elif args.model == "sonnet":
            response = send_to_anthropic_claude(designed_prompt)
            answer = response.strip().replace('"', '')
        elif args.model == "llama":
            response = send_to_llama(designed_prompt, pipeline=pipeline)
            answer = response.strip().replace('"', '')
        elif args.model == "gemini":
            response = send_to_google_gemini(designed_prompt)
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
        
    
        gt = get_temperature_diff_max_min(sample["output_window"])
        output = decode_temperature_indicator(answer)
        prediction = {
            "max": output[0],
            "min": output[1],
            "diff": output[2]
        }
        res = {
            "cnt": len(result_list),
            "filename": sample["filename"], 
            "ground_truth": gt, 
            "predict": prediction,
            "answer": answer
        }
        result_list.append(res)
        print("{}/{}: ground_truth: {}; predicted: {}.".format(idx, tot_samples, gt, prediction))
        
        metric_dict["diff"]["mse"].append(mean_squared_error([gt["diff"]], [prediction["diff"]]))
        metric_dict["diff"]["mae"].append(mean_absolute_error([gt["diff"]], [prediction["diff"]]))
        metric_dict["max"]["mse"].append(mean_squared_error([gt["max"]], [prediction["max"]]))
        metric_dict["max"]["mae"].append(mean_absolute_error([gt["max"]], [prediction["max"]]))
        metric_dict["min"]["mse"].append(mean_squared_error([gt["min"]], [prediction["min"]]))
        metric_dict["min"]["mae"].append(mean_absolute_error([gt["min"]], [prediction["min"]]))
    except Exception as e:
        # Handle the exception and print the error message
        print(f"An error occurred: {e}")
    
    if (idx +1) % 20 == 0:
        save_to_json(result_list, save_path=f"{args.save_path}/results.json")


summary_dict = {}
for key in metric_dict.keys():
    summary_dict[key] = {
        "mse": np.mean(metric_dict[key]["mse"]),
        "mae": np.mean(metric_dict[key]["mae"])
    }


save_to_json(result_list, save_path=f"{args.save_path}/results.json")
save_to_json(summary_dict, save_path=f"{args.save_path}/summary.json")
print(f"Processing complete. Results saved to {args.save_path}/summary.json")

