
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone


def plot_timeseries_with_reference(
    pairs,
    reference_timeseries=None,
    save_path="vis.png",
    stretch=None,
    event_points=None,
    event_descriptions=None,
):
    """
    Plot multiple input-output pairs with reference time series and event points.

    Parameters:
        pairs (list of tuples): List of pairs where each pair is 
            (input_window_timestamps, input_window, output_window_timestamps, output_window).
        reference_timeseries (tuple, optional): (reference_timestamps, reference_values).
        save_path (str, optional): Path to save the resulting plot.
        stretch (float, optional): Stretch the reference time series range by this factor.
        event_points (list of datetime, optional): List of event timestamps to mark on the plots.
        event_descriptions (list of str, optional): List of textual descriptions corresponding to event points.
    """
    # Set up the figure
    fig, axes = plt.subplots(
        nrows=2 if reference_timeseries else 1,
        ncols=1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 1]} if reference_timeseries else None
    )
    ax_main = axes[0] if reference_timeseries else axes
    ax_ref = axes[1] if reference_timeseries else None

    # Assign fixed random colors
    colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(pairs))]

    # Concatenate all timestamps for stretch calculation
    all_timestamps = []
    for input_ts, _, output_ts, _ in pairs:
        all_timestamps.extend(input_ts)
        all_timestamps.extend(output_ts)
    if stretch and reference_timeseries:
        center_time = min(all_timestamps) + (max(all_timestamps) - min(all_timestamps)) / 2
        time_range = (max(all_timestamps) - min(all_timestamps)) * stretch
        stretched_start = center_time - time_range / 2
        stretched_end = center_time + time_range / 2

    # Plot input-output pairs
    for i, (input_ts, input_vals, output_ts, output_vals) in enumerate(pairs):
        color = colors[i]
        ax_main.plot(
            input_ts, input_vals, label=f"Input {i+1}", color=color, linestyle="-"
        )
        ax_main.plot(
            output_ts, output_vals, label=f"Output {i+1}", color=color, linestyle="--"
        )

    # Mark event points on the main panel
    if event_points:
        for idx, event in enumerate(event_points):
            ax_main.axvline(event, color="red", linestyle=":", alpha=0.7, label=f"Event {idx+1}")


    ax_main.set_title("Input-Output Time Series")
    ax_main.set_xlabel("Time")
    ax_main.set_ylabel("Values")
    ax_main.legend()
    ax_main.grid()

    # Plot reference time series if provided
    if reference_timeseries:
        ref_ts, ref_vals = reference_timeseries
        ax_ref.plot(ref_ts, ref_vals, color="gray", linestyle="-", alpha=0.7)

        # Add bounding boxes for each pair
        for i, (input_ts, _, output_ts, _) in enumerate(pairs):
            color = colors[i]
            start = min(input_ts + output_ts)
            end = max(input_ts + output_ts)
            ax_ref.axvspan(start, end, color=color, alpha=0.3, label=f"Pair {i+1} Window")

        # Mark event points on the reference panel
        if event_points:
            for idx, event in enumerate(event_points):
                ax_ref.axvline(event, color="red", linestyle=":", alpha=0.7)

        if stretch:
            ax_ref.set_xlim(stretched_start, stretched_end)
            ax_ref.set_ylim(ax_main.get_ylim())
            
        ax_ref.set_title("Reference Time Series")
        ax_ref.set_xlabel("Time")
        ax_ref.set_ylabel("Reference Values")
        ax_ref.legend()
        ax_ref.grid()

    # Format x-axis for datetime
    for ax in ([ax_main, ax_ref] if reference_timeseries else [ax_main]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add textual descriptions as a legend or annotation
    if event_descriptions:
        description_text = "\n".join([f"Event {i+1}: {desc}" for i, desc in enumerate(event_descriptions)])
        plt.figtext(0.5, 0.01, description_text, ha="center", fontsize=10, color="red")

    # Save or show plot
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        # print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def sample_a_stock_pair_flexible_units_given_event(
    timestamps_ms,
    values,
    event_timestamp_ms,
    event_position_ratio=0.5,
    input_window_size=1,
    input_window_unit="days",
    input_granularity=1,
    input_granularity_unit="hours",
    output_window_size=1,
    output_window_unit="days",
    output_granularity=1,
    output_granularity_unit="hours"
):
    """
    Creates a time series prediction pair based on a given timestamp and parameters.
    
    Parameters:
        timestamps_ms (list): Entire list of timestamps in milliseconds.
        values (list): Entire list of associated values.
        event_timestamp_ms (int): Specific timestamp in milliseconds for the event.
        event_position_ratio (float): Relative position of the event (0-1) within the input range.
        input_window_size (int): Size of the input window.
        input_window_unit (str): Unit of the input window ("days" or "hours").
        input_granularity (int): Granularity of the input window.
        input_granularity_unit (str): Unit of input granularity ("days" or "hours").
        output_window_size (int): Size of the output window.
        output_window_unit (str): Unit of the output window ("days" or "hours").
        output_granularity (int): Granularity of the output window.
        output_granularity_unit (str): Unit of output granularity ("days" or "hours").
    
    Returns:
        tuple: A tuple containing:
            - input_timestamp_list: List of input timestamps.
            - input_value_list: List of input values.
            - output_timestamp_list: List of output timestamps.
            - output_value_list: List of output values.
    """
    # Convert milliseconds to datetime
    timestamps = [datetime.utcfromtimestamp(ts / 1000.0) for ts in timestamps_ms]

    # Determine the event timestamp
    if event_timestamp_ms is None:
        # Calculate the event timestamp based on the ratio
        index = int(event_position_ratio * (len(timestamps) - 1))
        event_timestamp = timestamps[index]
    else:
        # Convert provided timestamp from ms to datetime
        event_timestamp = datetime.utcfromtimestamp(event_timestamp_ms / 1000.0)

    # Helper function to convert size and unit to timedelta
    def to_timedelta(size, unit):
        if unit == "days":
            return timedelta(days=size)
        elif unit == "hours":
            return timedelta(hours=size)
        elif unit == "minutes":
            return timedelta(minutes=size)
        elif unit == "seconds":
            return timedelta(seconds=size)
        elif unit == "weeks":
            return timedelta(weeks=size)
        elif unit == "months":
            return timedelta(months=size)
        else:
            raise ValueError("Invalid unit. Use 'days' or 'hours'.")

    # Define input and output windows
    input_window_start = event_timestamp - to_timedelta(input_window_size, input_window_unit)
    input_window_end = event_timestamp
    output_window_start = event_timestamp
    output_window_end = event_timestamp + to_timedelta(output_window_size, output_window_unit)

    sample_every = int(to_timedelta(input_granularity, input_granularity_unit).total_seconds() / (5*60))
    # Sample input timestamps and values
    input_timestamps = [
        ts for ts in timestamps if input_window_start <= ts < input_window_end
    ][::sample_every]
    input_values = [
        values[i] for i, ts in enumerate(timestamps) if ts in input_timestamps
    ]

    # Sample output timestamps and values
    output_timestamps = [
        ts for ts in timestamps if output_window_start <= ts < output_window_end
    ][::int(to_timedelta(output_granularity, output_granularity_unit).total_seconds() / (5*60))]
    output_values = [
        values[i] for i, ts in enumerate(timestamps) if ts in output_timestamps
    ]

    return {
        "input_timestamps":[x.timestamp() for x in input_timestamps],
        "input_window":input_values,
        "output_timestamps": [x.timestamp() for x in output_timestamps],
        "output_window": output_values
    }

def calculate_stock_trend(
    input_timestamps,
    input_prices,
    output_timestamps,
    output_prices,
    statistic="mean",
    bins=None,
    auto_bin_method="quantile",
    auto_bin_count=5,
    labels=None
):
    """
    Calculate stock trends for input, output, and overall periods.

    Parameters:
        input_timestamps (list): List of input timestamps.
        input_prices (list): List of input stock prices.
        output_timestamps (list): List of output timestamps.
        output_prices (list): List of output stock prices.
        statistic (str): Statistic to calculate ("mean", "last", "median", "std").
        bins (list): Custom bin edges for categorizing percentage changes.
        auto_bin_method (str): Method for automatic bin creation ("quantile", "uniform").
        auto_bin_count (int): Number of bins to create if bins are auto-generated.
        labels (list): Labels for bins.

    Returns:
        dict: Contains:
            - "input_percentage_change": Percentage change within the input period.
            - "output_percentage_change": Percentage change within the output period.
            - "overall_percentage_change": Percentage change from the start of the input period to the end of the output period.
            - "input_bin_label": Bin label for input period change.
            - "output_bin_label": Bin label for output period change.
            - "overall_bin_label": Bin label for overall period change.
    """

    def calculate_stat(prices, stat_type):
        """Helper function to calculate statistic."""
        if stat_type == "mean":
            return np.mean(prices).item()
        elif stat_type == "last":
            return prices[-1]
        elif stat_type == "median":
            return np.median(prices).item()
        elif stat_type == "std":
            return np.std(prices).item()
        else:
            raise ValueError("Unsupported statistic. Choose from 'mean', 'last', 'median', 'std'.")

    # Calculate statistics for input, output, and overall prices
    input_stat = calculate_stat(input_prices, statistic)
    output_stat = calculate_stat(output_prices, statistic)
    overall_stat_start = input_prices[0]  # First price in the input period
    overall_stat_end = output_prices[-1]  # Last price in the output period

    # Compute percentage changes
    input_percentage_change = ((input_stat - input_prices[0]) / input_prices[0]) * 100
    output_percentage_change = ((output_stat - output_prices[0]) / output_prices[0]) * 100
    overall_percentage_change = ((overall_stat_end - overall_stat_start) / overall_stat_start) * 100

    # Determine bins if not provided
    if bins is None:
        if auto_bin_method == "quantile":
            data = [input_percentage_change, output_percentage_change, overall_percentage_change]
            bins = pd.qcut(data, q=auto_bin_count, retbins=True, duplicates="drop")[1]
        elif auto_bin_method == "uniform":
            min_val, max_val = -50, 50  # Default bin range
            bins = np.linspace(min_val, max_val, auto_bin_count + 1)
        else:
            raise ValueError("Unsupported auto_bin_method. Choose from 'quantile', 'uniform'.")

    # Generate bin labels if not provided
    if labels is None:
        labels = [f"Bin {i+1}" for i in range(len(bins) - 1)]

    # Assign bin labels
    input_bin_label = pd.cut([input_percentage_change], bins=bins, labels=labels, include_lowest=True)[0]
    output_bin_label = pd.cut([output_percentage_change], bins=bins, labels=labels, include_lowest=True)[0]
    overall_bin_label = pd.cut([overall_percentage_change], bins=bins, labels=labels, include_lowest=True)[0]

    return {
        "input_percentage_change": input_percentage_change,
        "input_bin_label": input_bin_label,
        "output_percentage_change": output_percentage_change,
        "output_bin_label": output_bin_label,
        "overall_percentage_change": overall_percentage_change,
        "overall_bin_label": overall_bin_label,
    }

def save_dict_to_local_json(dictionary, file_path):
    """
    Save a dictionary to a JSON file at the given path.

    Args:
        dictionary (dict): The dictionary to save.
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)
        # print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary: {e}")

def read_json_from_local(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_folder_of_json(folder_path):
    json_data_list = []
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path {folder_path} is not a directory.")
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            try:
                json_data = read_json_from_local(file_path)
                json_data_list.append({"file_name": file_name, "data": json_data})
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    
    return json_data_list

def convert_iso8601_to_datetime_and_timestamp(iso8601_string):
    """
    Convert an ISO 8601 string to a datetime object and a timestamp in nanoseconds.

    Args:
        iso8601_string (str): The ISO 8601 formatted string (e.g., '2023-09-06T22:34:00Z').

    Returns:
        tuple: A tuple containing:
            - datetime object
            - timestamp in nanoseconds (int)
    """
    # Parse the string into a datetime object
    dt = datetime.strptime(iso8601_string, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    # Convert to timestamp in nanoseconds
    timestamp_ns = int(dt.timestamp() * 1e3)

    return dt, timestamp_ns

def calculate_moving_average(prices, window, ma_type="SMA"):
    """
    Calculate SMA or EMA for a given price series and window size.

    Parameters:
        prices (list): List of stock prices.
        window (int): Window size for moving average (e.g., 10, 50, 200).
        ma_type (str): Type of moving average, "SMA" or "EMA".

    Returns:
        dict: Contains calculated SMA or EMA with key formatted as "{type}-{window}".
    """
    if len(prices) < window:
        return {f"{ma_type}-{window}": None, "error": f"Insufficient data for {ma_type}-{window}"}

    prices_series = pd.Series(prices)

    if ma_type == "SMA":
        ma_value = prices_series.rolling(window=window, min_periods=1).mean().iloc[-1]
    elif ma_type == "EMA":
        ma_value = prices_series.ewm(span=window, adjust=False).mean().iloc[-1]
    else:
        raise ValueError("ma_type must be 'SMA' or 'EMA'")

    return {f"{ma_type}-{window}": ma_value}

# Function to compute simple moving averages (SMA)
def compute_sma(prices, window):
    return pd.Series(prices).rolling(window=window, min_periods=1).mean().tolist()

# Function to compute exponential moving averages (EMA)
def compute_ema(prices, window):
    return pd.Series(prices).ewm(span=window, adjust=False).mean().tolist()

# Function to compute MACD and Signal Line
def compute_macd(prices, short_window=12, long_window=26, signal_window=9):
    ema_short = pd.Series(prices).ewm(span=short_window, adjust=False).mean()
    ema_long = pd.Series(prices).ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd.tolist(), signal_line.tolist()

# Function to compute Bollinger Bands
def compute_bollinger_bands(prices, window=20):
    series = pd.Series(prices)
    sma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    upper_band = (sma + (2 * std)).tolist()
    lower_band = (sma - (2 * std)).tolist()
    return upper_band, lower_band