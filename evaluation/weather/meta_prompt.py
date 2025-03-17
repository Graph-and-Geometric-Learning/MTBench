import re
from typing import List

def temperature_trend_metaprompt_generation(
    text: str,
    past_temperatures: List[float],
    start_datetime: str,
    end_datetime: str,
    granularity: str,
    past_days: int,
    next_days: int,
    mode: str,
    past_future: str
) -> str:
    """
    Generates a meta-prompt for temperature prediction based on given inputs.

    Args:
        text (str): News article content within the input time series range.
        temperatures (List[float]): Historical temperature data.
        start_datetime (str): Start datetime of the input time series.
        end_datetime (str): End datetime of the input time series.
        granularity (str): Granularity of the input time series (e.g., hourly, daily).
        past_days (int): Number of past days to analyze.   
        next_days (int): Number of future days to predict.
        mode (str): Mode of prediction ("timeseries_only", "text_only", "combined").
        past_future (str): Whether to predict the past or future trend ("past", "future").
        
    Returns:
        str: Meta-prompt for ChatGPT.
    """
    mode = mode.lower()
    head1=f"You are a weather forecasting AI. Your task is to analyze the past {past_days} days's of temperature trend"
    head2=f" and predict the temperature trend for the next {next_days} days'."
    if past_future=="future":
        head=head1+" "+head2
    else:
        head=head1
    prompt = (
        f"The input time series represents temperature readings "
        f"from {start_datetime} to {end_datetime}, with a granularity of {granularity}. "
        f"This data is from a location in the United States, where summers are hot and winters are cold. "
        f"Weather conditions can also be affected by storms, heavy rain, and cold fronts."
        f"The daytime is usually warmer than the nighttime."
        f"Every 24 temperature readings represent a full day from 00:00 to 23:00."
    )
    prompt=head+prompt
    
    if mode == "timeseries_only":
        prompt += (
            "\n\nYou will use only the provided historical temperature data to make your prediction. "
            f"The input temperature readings are: {past_temperatures}. "
        )
    elif mode == "text_only":
        prompt += (
            "\n\nYou will use only the provided news article content to make your prediction. "
            f"The news article states: {text}. "
        )
    elif mode == "combined":
        prompt += (
            "\n\nYou will use both the historical temperature data and the news article to make your prediction. "
            f"The input temperature readings are: {past_temperatures}. "
            f"The news article states: {text}. "
        )
    else:
        raise ValueError(
            "Invalid mode. Choose from 'timeseries_only', 'text_only', or 'combined'."
        )

    
    if past_future=="future":
        prompt += (
            f"\n\nBased on the information you received, predict the temperature trend for the next {next_days} days. Calculate the mean temperature of the last 24-hour period (i.e., the most recent day in the input) and compare it with the mean temperature of the first predicted day. If the difference is greater or equal than 0.5, classify the trend as 'increasing'. If the difference is less or equal than -0.5, classify the trend as 'decreasing'. Otherwise, classify it as 'stable'. Return one word in increasing, decreasing, or stable, without any reasoning text or extra words." 
        )
    else:
        prompt += (
        "\n\nCompute the daily mean temperature and fit a linear trend, if the slope is greater or equal to 0.25, then it is classified as increasing, if the slope is less or equal than -0.25, it is classified as decrease, if the slope is between 0.25 to -0.25, it is classified as stable."
        
        
    )
        prompt += (
            f"\n\nBased on the trend, predict the temperature trend for the past {past_days} days. Return one word in increasing, decreasing, or stable. Do not return reasoning text and extra words."
        )
    return prompt

def temperature_forecast_metaprompt_generation(
    text: str,
    temperatures: List[float],
    start_datetime: str,
    end_datetime: str,
    granularity: str,
    prediction_length: int,
    mode: str,
    next_days: int = 1
) -> str:
    """
    Generates a meta-prompt for temperature forecasting based on given inputs.

    Args:
        text (str): News article content within the input time series range.
        temperatures (List[float]): Historical temperature data.
        start_datetime (str): Start datetime of the input time series.
        end_datetime (str): End datetime of the input time series.
        granularity (str): Granularity of the input time series (e.g., hourly, daily).
        prediction_length (int): Number of future time steps to predict.
        mode (str): Mode of prediction ("timeseries_only", "text_only", "combined").

    Returns:
        str: Meta-prompt for ChatGPT.
    """
    mode = mode.lower()
    prompt = (f"You are a weather forecasting AI. Your task is to predict the next {prediction_length} time steps "
        f"for temperature based on the given data. The input time series represents temperature readings "
        f"from {start_datetime} to {end_datetime}, with a granularity of {granularity}. "
        f"The prediction should cover the period for next {next_days} days "
        f"with the same granularity ({granularity}). \n\n")
    
    prompt += (
        f"This data is from a location in the United States, where summers are hot and winters are cold. "
        f"Weather conditions can also be affected by storms, heavy rain, and cold fronts."
        f"The daytime is usually warmer than the nighttime."
        f"Every 24 temperature readings represent a full day from 00:00 to 23:00."
    )

    if mode == "timeseries_only":
        prompt += (
            "\n\nYou will use only the provided historical temperature data to make your prediction. "
            f"The input temperature readings are: {temperatures}. "
        )
    elif mode == "text_only":
        prompt += (
            "\n\nYou will use only the provided news article content to make your prediction. "
            f"The news article states: {text}. "
        )
    elif mode == "combined":
        prompt += (
            "\n\nYou will use both the historical temperature data and the news article to make your prediction. "
            f"The input temperature readings are: {temperatures}. "
            f"The news article states: {text}. "
        )
    else:
        raise ValueError(
            "Invalid mode. Choose from 'timeseries_only', 'text_only', or 'combined'."
        )

    prompt += (
        "\n\nReturn your prediction as a list of float values in plain text, strictly following this format:\n"
        "Predicted Temperatures: value1, value2, ..., valueN."
        "Ensure no extra text or explanations are included. "
        f"Make sure the length of the predicted temperatures is exactly equal to {prediction_length}. And each value should be rounded to 2 decimal places."
    )
    return prompt

def temperature_indicator_metaprompt_generation(
    text: str,
    past_temperatures: List[float],
    start_datetime: str,
    end_datetime: str,
    granularity: str,
    past_days: int,
    mode: str,
    next_days: int = 1
) -> str:
    mode = mode.lower()
    
    prompt=f"You are a weather forecasting AI. Your task is to analyze the past {past_days} days's of temperature trend and predict the next {next_days} days's highest temperature and lowest temperature as well as the temperature difference between the highest and lowest temperature based on the given data."
    
    prompt += (
        f"The input time series represents temperature readings "
        f"from {start_datetime} to {end_datetime}, with a granularity of {granularity}. "
        f"This data is from a location in the United States, where summers are hot and winters are cold."
        f"Weather conditions can also be affected by storms, heavy rain, and cold fronts."
        f"The daytime is usually warmer than the nighttime."
        f"Every 24 temperature readings represent a full day from 00:00 to 23:00."
    )
    if mode == "timeseries_only":
        prompt += (
            "\n\nYou will use only the provided historical temperature data to make your prediction. "
            f"The input temperature readings are: {past_temperatures}. "
        )
    elif mode == "text_only":
        prompt += (
            "\n\nYou will use only the provided news article content to make your prediction. "
            f"The news article states: {text}. "
        )
    elif mode == "combined":
        prompt += (
            "\n\nYou will use both the historical temperature data and the news article to make your prediction. "
            f"The input temperature readings are: {past_temperatures}. "
            f"The news article states: {text}. "
        )
    else:
        raise ValueError(
            "Invalid mode. Choose from 'timeseries_only', 'text_only', or 'combined'."
        )
    prompt += (
            f"\n\n Now, you need to predict the next {next_days} days's highest temperature, lowest temperature and temperature difference between the highest and lowest temperature based on the data provided. Your response should be in the format:'Highest temperature: , Lowest temperature: , Temperature difference: ' without extra analysis and other words" 
        )
    return prompt

def weather_mcqa_metaprompt_generation(
    past_days: int,
    start_datetime: str,
    end_datetime: str,
    past_temperatures: List[float],
    news: str,
    question: str
) -> str:

    system_prompt =f"You have a {past_days}-day temperature time series, a weather event report published on the last day of time series. Answer the questions, return your answer in single letter (A, B, C, D) without other words."
        
    query = f"{past_days}-day temperature time series between{start_datetime} to {end_datetime}, time interval is 1 hour:\
            {past_temperatures}\
            weather event report: \
            {news}\
            {question} Answer:"
    prompt = f"{system_prompt}\n\n{query}"
    return prompt



def decode_temperature_indicator(output: str):
    output = output.strip().lower()

    # Regular expression to match both integer and float values
    pattern = r"highest temperature:\s*(-?\d+\.?\d*)\s*,?\s*lowest temperature:\s*(-?\d+\.?\d*)\s*,?\s*temperature difference:\s*(-?\d+\.?\d*)\s*"
    
    match = re.fullmatch(pattern, output)
    if not match:
        raise ValueError(f"Invalid output format. The output is: {output}")

    highest_temp, lowest_temp, temp_diff = map(float, match.groups())

    # Ensure that the temperature difference is correctly calculated
    expected_diff = round(highest_temp - lowest_temp, 2)  # Rounding to avoid floating-point precision issues
    if expected_diff != round(temp_diff, 2):
        raise ValueError(f"Inconsistent temperature difference. The output is: {output}")

    return highest_temp, lowest_temp, temp_diff


def decode_temperature_trend_prediction(output: str):
    output = output.strip().lower()
    trend_words = ["increasing", "decreasing", "stable"]
    
    # Find all matching trend words in the output
    found_trends = [word for word in trend_words if word in output]
    
    # Check if exactly one trend word was found
    if len(found_trends) == 0:
        raise ValueError(f"No decision word found. The output is: {output}")
    elif len(found_trends) > 1:
        raise ValueError(f"More than one decision word found. The output is: {output}")
    
    # Return the one found trend word
    return found_trends[0]

def decode_temperature_forecast(response: str) -> List[float]:
    """
    Decodes the response from ChatGPT to extract the predicted temperatures.

    Args:
        response (str): The raw response from ChatGPT.
        expected_length (int): Expected number of predicted time steps.

    Returns:
        List[float]: Extracted list of predicted temperatures.
    """
    match = re.search(r"Predicted Temperatures:\s*([-?\d.,\s]+)", response, re.IGNORECASE)
    if match:
        try:
            temperatures_str = match.group(1)
            temperatures = []
            for temp in temperatures_str.split(","):
                temp_cleaned = re.sub(r"\.$", "", temp.strip())  # Only removes trailing dot(s)
                temperatures.append(float(temp_cleaned))
            
    
            # if len(temperatures) != expected_length:
            #     raise ValueError(f"Expected {expected_length} temperatures, but got {len(temperatures)}. Response: {response}")

            return temperatures
        except Exception as e:
            raise ValueError(f"Error parsing temperature predictions: {e}. Response: {response}")
    else:
        raise ValueError(f"Could not parse the predicted temperatures. Response: {response}")