import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append("../..")
import os
from meta_prompt import  weather_mcqa_metaprompt_generation
from evaluation.utils import (
    save_to_json,
    calculate_mcqa_acc
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
parser.add_argument("--setting", type=str, default="long", help="weather_mcqa")
args = parser.parse_args()
args.in_days = 14 if args.setting == "long" else 7
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
            "text": data.get("text"),
            "input_timestamps": data.get("input_timestamps"),
            "question": data.get("question"),
            "answer": data.get("answer")
        }
        data_list.append(extracted_data)

# data_list = data_list[:10]  # for testing

os.makedirs(Path(args.save_path).parent, exist_ok=True)
result_list = []
tot_samples = len(data_list)
print("Evaluating {} samples......".format(tot_samples))

for idx, sample in enumerate(data_list):
    filename = sample["filename"]
    
    designed_prompt = weather_mcqa_metaprompt_generation(
        past_days=args.in_days,
        start_datetime=sample["input_timestamps"][0],
        end_datetime=sample["input_timestamps"][-1],
        past_temperatures=sample["input_window"],
        news=sample["text"],
        question=sample["question"]
    )
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
        
        res = {
            "cnt": len(result_list),
            "filename": sample["filename"], 
            "question": sample["question"],
            "ground_truth": sample["answer"], 
            "predict": answer,
        }
        result_list.append(res)

    except Exception as e:
        # Handle the exception and print the error message
        print(f"An error occurred: {e}")
    
    if (idx +1) % 20 == 0:
        save_to_json(result_list, save_path=f"{args.save_path}/results.json")


save_to_json(result_list, save_path=f"{args.save_path}/results.json")
accuracy = calculate_mcqa_acc(result_list)
print(f"Accuracy: {accuracy}%")
print(f"Processing complete. Results saved to {args.save_path}/results.json")


