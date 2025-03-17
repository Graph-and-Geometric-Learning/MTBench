import numpy as np
import pandas as pd
import os
import json
import re
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
def remove_duplicate_sentences(paragraph):
    # Step 1: Split the paragraph into sentences
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)  # Split on punctuation followed by whitespace
    
    # Step 2: Remove duplicate sentences while preserving order
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    # Step 3: Reconstruct the paragraph
    cleaned_paragraph = ' '.join(unique_sentences)
    return cleaned_paragraph
def extract_temperature_data(timestamp, numbers_data,previous_days,after_days):
    """
    Extract temperature data for the range [timestamp's date - 6 days, timestamp's date] and 
    the entire day after timestamp's date.
    
    Parameters:
        timestamp (pd.Timestamp): The center timestamp.
        numbers_data (pd.DataFrame): The DataFrame containing weather data.
    
    Returns:
        tuple: Two lists - (temperatures before, temperatures after, dates before, dates after).
    """
    # Convert timestamp to date (removes time component)
    target_date = timestamp.date()
    
    # Define the calendar day ranges
    before_start = target_date - pd.Timedelta(days=previous_days-1)  # 6 days before + current day = 7 days total
    after_end = target_date+pd.Timedelta(days=after_days)  # The full day after
    
    # Extract temperatures and corresponding dates for the 7 full calendar days before (including the current day)
    mask_before = (numbers_data['DATE'].dt.date >= before_start) & (numbers_data['DATE'].dt.date <= target_date)
    data_before = numbers_data.loc[mask_before, "DATE"].tolist()
    temp_before = numbers_data.loc[mask_before, 'temperature'].tolist()

    # Extract temperatures and corresponding dates for the full calendar day after
    mask_after = (numbers_data['DATE'].dt.date>target_date) & (numbers_data['DATE'].dt.date<=after_end)
    data_after = numbers_data.loc[mask_after, "DATE"].tolist()
    temp_after = numbers_data.loc[mask_after, 'temperature'].tolist()
    
    return temp_before, temp_after, data_before, data_after
def interpolate_temperature_data(data_before, temp_before, data_after, temp_after,day_before,day_after):
    """
    Ensures that the past n days (n*24 hourly points) and next m day (n m24 hourly points) contain 
    exactly the required number of hourly data points by resampling and interpolating missing values.

    Parameters:
        data_before (list): List of timestamps for the previous n days.
        temp_before (list): List of temperatures for the previous n days.
        data_after (list): List of timestamps for the next m day.
        temp_after (list): List of temperatures for the next m day.

    Returns:
        tuple: (Interpolated timestamps, Interpolated temperatures)
    """
    
    def interpolate_data(data, temp, hours):
        """
        Generic function to interpolate temperature data over a specified number of hours.

        Parameters:
            data (list): List of timestamps.
            temp (list): List of temperatures.
            hours (int): Total hours to interpolate.

        Returns:
            tuple: (Interpolated timestamps, Interpolated temperatures)
        """
        if not data or not temp:
            return [], []  # No data to interpolate

        # Convert lists to DataFrame
        df = pd.DataFrame({'DATE': data, 'temperature': temp})
        df['DATE'] = pd.to_datetime(df['DATE'])  # Ensure datetime format
        df.set_index('DATE', inplace=True)  # Set index for resampling

        # Sort by time to avoid issues
        df = df.sort_index()

        # Generate full range of hourly timestamps
        full_hours = pd.date_range(start=df.index.min(), periods=hours, freq='h')

        # Create a DataFrame with the full range of hours
        full_day_df = pd.DataFrame({'DATE': full_hours})
        full_day_df.set_index('DATE', inplace=True)

        # Merge available data
        merged_data = full_day_df.merge(df, left_index=True, right_index=True, how="left")

        # Interpolate missing values
        merged_data['temperature'] = merged_data['temperature'].interpolate(method='time')

        # Ensure no remaining NaN values (forward-fill and backward-fill as a backup)
        merged_data['temperature'] = merged_data['temperature'].ffill().bfill()

        return merged_data.index.tolist(), merged_data["temperature"].tolist()

    # Interpolate past 7 days (168 hours)
    timestamps_before, temps_before = interpolate_data(data_before, temp_before, day_before * 24)

    # Interpolate next day (24 hours)
    timestamps_after, temps_after = interpolate_data(data_after, temp_after, day_after*24)


    return timestamps_before, temps_before, timestamps_after, temps_after
def generate_QA_data(station_id,top_number,days_before,days_after,QA_path,news_path,time_series_path):
    os.makedirs(QA_path,exist_ok=True)
    sub_path=os.path.join(QA_path,str(days_before)+"_"+str(days_after))
    
    sub_path = os.path.join(sub_path, str(top_number))
    sub_path = os.path.join(sub_path, station_id)
    os.makedirs(sub_path, exist_ok=True)

    news_data = pd.read_json(os.path.join(news_path, "news_" + station_id + ".json"))
    numbers_data = pd.read_json(os.path.join(time_series_path, station_id + "_processed.json"))

    # Convert timestamps to datetime
    news_data["BEGIN_TIMESTAMP"] = pd.to_datetime(news_data["BEGIN_TIMESTAMP"])
    news_data["END_TIMESTAMP"] = pd.to_datetime(news_data["END_TIMESTAMP"])
    numbers_data["DATE"] = pd.to_datetime(numbers_data["DATE"])

    # Process news text
    news_data["NEWS"] = news_data["NEWS"].apply(remove_duplicate_sentences)

    # Sort by longest news entries
    news_data['NEWS_LENGTH'] = news_data['NEWS'].str.len()
    news_data_sorted = news_data.sort_values(by='NEWS_LENGTH', ascending=False)
    top_longest_news = news_data_sorted.head(top_number)
    temp_data = top_longest_news['END_TIMESTAMP'].apply(
        lambda x: extract_temperature_data(x,numbers_data,previous_days=days_before,after_days=days_after)  # Ensure full event day included
    )
    top_longest_news.loc[:, 'TEMP_BEFORE'] = temp_data.apply(lambda x: x[0])
    top_longest_news.loc[:, 'TEMP_AFTER'] = temp_data.apply(lambda x: x[1])
    top_longest_news.loc[:, 'DATE_BEFORE'] = temp_data.apply(lambda x: x[2])
    top_longest_news.loc[:, 'DATE_AFTER'] = temp_data.apply(lambda x: x[3])
    interpolated_data = top_longest_news.apply(
        lambda row: interpolate_temperature_data(
            row['DATE_BEFORE'], row['TEMP_BEFORE'],
            row['DATE_AFTER'], row['TEMP_AFTER'],day_before=days_before,day_after=days_after
        ),axis=1
    )
    top_longest_news.loc[:, 'TEMP_BEFORE_Inter'] = interpolated_data.apply(lambda x: x[1])
    top_longest_news.loc[:, 'TEMP_AFTER_Inter'] = interpolated_data.apply(lambda x: x[3])
    top_longest_news.loc[:, 'DATE_BEFORE_Inter'] = interpolated_data.apply(lambda x: x[0])
    top_longest_news.loc[:, 'DATE_AFTER_Inter'] = interpolated_data.apply(lambda x: x[2])

    
    for i in range(top_longest_news.shape[0]):
        entry = top_longest_news.iloc[i]

        data_dict = {
                    "text": remove_duplicate_sentences(entry["NEWS"]),
                    "input_timestamps": [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in entry["DATE_BEFORE_Inter"]],
                    "input_window": entry["TEMP_BEFORE_Inter"],
                    
                    "output_timestamps": [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in entry["DATE_AFTER_Inter"]],
                    "output_window": entry["TEMP_AFTER_Inter"]  # Interpolated to 24 hourly points
                }
        with open(os.path.join(sub_path, f"{i}.json"), "w") as f:
                json.dump(data_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms", type=str, default="14_3", help="Comma-separated pairs of days_before and days_after (e.g., '14_3,7_2')")
    parser.add_argument("--QA_path", type=str, default="QA", help="Path for QA output")
    parser.add_argument("--top_numbers", type=str, default="40", help="Comma-separated list of top numbers (e.g., '40')")
    parser.add_argument("--news_path", type=str, default="news_data", help="Path to news data")
    parser.add_argument("--time_series_path", type=str, default="time_series_data", help="Path to time series data")
    parser.add_argument("--station_file", type=str, default="stations.csv", help="Path to station ID file")
    args = parser.parse_args()
    
    terms = [tuple(map(int, term.split('_'))) for term in args.terms.split(',')]
    top_numbers = list(map(int, args.top_numbers.split(',')))
    
    id_list = pd.read_csv(args.station_file)["id"].tolist()
    for days_before, days_after in terms:
        for top_number in top_numbers:
            for station_id in tqdm(id_list):
                generate_QA_data(station_id, top_number, days_before, days_after, args.QA_path, args.news_path, args.time_series_path)

