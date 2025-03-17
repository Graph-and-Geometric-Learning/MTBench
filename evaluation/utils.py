import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression

def format_time_difference(seconds):
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24

    if days > 0:
        if hours % 24 > 0.1:
            return f"{days} days-{hours % 24} hours"
        else:
            return f"{days} days"
    elif hours > 0:
        if minutes % 60 > 0.1:
            return f"{hours} hours-{minutes % 60} minutes"
        else:
            return f"{hours} hours"
    elif minutes > 0:
        if seconds % 60 > 0.1:
            return f"{minutes} minutes-{seconds % 60} seconds"
        else:
            return f"{minutes} minutes"
    else:
        return f"{seconds} seconds"

def save_to_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def plot_series(filename, input_ts, output_ts, predicted_ts, save_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(input_ts)), input_ts, label="Input Time Series", marker='o')
    plt.plot(range(len(input_ts), len(input_ts) + len(output_ts)), output_ts, label="Ground Truth", marker='o')
    plt.plot(range(len(input_ts), len(input_ts) + len(predicted_ts)), predicted_ts, label="Predicted", linestyle='dashed')
    plt.legend()
    plt.title(f"Prediction for {filename}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig(os.path.join(save_folder, filename.replace('.json', '.png')))
    plt.close()

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_acc(result_list, regrouped_labels = None):
    if regrouped_labels is None:
        correct_pred = sum(1 for result in result_list if result["ground_truth"] in result["predict"])
    else:
        correct_pred = 0
        for result in result_list:
            gt_group = regrouped_labels[result['ground_truth']]
            for original_label in regrouped_labels.keys():
                if original_label in result['predict']:
                    predict_group = regrouped_labels[original_label]
                    if gt_group == predict_group:
                        correct_pred += 1
                        break

    total_pred = len(result_list)
    accuracy = correct_pred / total_pred 
                
    return accuracy


def calculate_correlation_acc(result_list):
    model_predictions = {"total": 0, "exact_correct": 0, "brief_correct": 0}
    positive_correlations = ["Strong Positive Correlation", "Moderate Positive Correlation"]
    negative_correlations = ["Strong Negative Correlation", "Moderate Negative Correlation"]
    for result in result_list:
        prediction = result["predict"].strip()
        model_predictions["total"] += 1
        if prediction == result["ground_truth"]:
            model_predictions["exact_correct"] += 1
        
        # Brief accuracy
        pred_is_positive = prediction in positive_correlations
        pred_is_negative = prediction in negative_correlations
        truth_is_positive = result["ground_truth"] in positive_correlations
        truth_is_negative = result["ground_truth"] in negative_correlations
        
        if (pred_is_positive and truth_is_positive) or \
            (pred_is_negative and truth_is_negative) or \
            (prediction == result["ground_truth"]):      
            model_predictions["brief_correct"] += 1
    
    # Calculate and format results
    total = model_predictions["total"]
    exact_accuracy = (model_predictions["exact_correct"] / total) * 100
    brief_accuracy = (model_predictions["brief_correct"] / total) * 100
        
    metric_results = {
        "exact_accuracy": f"{round(exact_accuracy, 2)}%",
        "brief_accuracy": f"{round(brief_accuracy, 2)}%",
        "total_samples": total
    }
    return metric_results


def calculate_mcqa_acc(result_list):
    correct = 0
    total = 0
    for result in result_list:
        predition = result["predict"].strip()
        predition = predition[0].upper()
        if predition == result["ground_truth"]:
            correct += 1
            
        total += 1
            
    accuracy = correct / total
    
    return accuracy * 100



def compute_temperature_trend(past_temperatures, next_day_temperatures=None):
    if next_day_temperatures is None:
        past_temperatures = np.array(past_temperatures).reshape(len(past_temperatures)//24, 24)
        daily_mean = np.mean(past_temperatures, axis=1)
        daily_mean = daily_mean.reshape(-1, 1)
        x = np.arange(len(daily_mean)).reshape(-1, 1)
        reg = LinearRegression().fit(x, daily_mean)
        slope = reg.coef_[0][0]
        if slope >= 0.25:
            return "increasing"
        elif slope <= -0.25:
            return "decreasing"
        else:
            return "stable"
    else:
        past_temperatures = np.array(past_temperatures).reshape(len(past_temperatures)//24, 24)
        ## last 24 hours
        last_day_temp = past_temperatures[-1]
        last_day_mean = np.mean(last_day_temp)
        next_day_mean = np.mean(next_day_temperatures)
        diff = next_day_mean - last_day_mean
        if diff >= 0.5:
            return "increasing"
        elif diff <= -0.5:
            return "decreasing"
        else:
            return "stable"
        
        
def get_temperature_diff_max_min(temperatures):
    temperatures = np.array(temperatures).reshape(len(temperatures)//24, 24)
    daily_max = float(np.max(temperatures, axis=1)[0])
    daily_min = float(np.min(temperatures, axis=1)[0]) 
    diff = round(daily_max - daily_min, 1)
    return {"max": daily_max, "min": daily_min, "diff": diff}
