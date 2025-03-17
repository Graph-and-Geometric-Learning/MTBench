import re
from typing import List, Union

def finance_mse_metaprompt_generation(
    text: str,
    prices: List[float],
    start_datetime: str,
    end_datetime: str,
    pred_end_datetime: str,
    granularity: str,
    prediction_length: int,
    mode: str,
) -> str:
    """
    Generates a meta-prompt for hypothetical stock price trend analysis 
    based on given inputs.

    Args:
        text (str): News article content within the input time series range.
        prices (List[float]): Historical stock prices.
        start_datetime (str): Start datetime of the input time series.
        end_datetime (str): End datetime of the input time series.
        pred_end_datetime (str): End datetime of the hypothetical projection.
        granularity (str): Granularity of the input time series (e.g., daily, hourly).
        prediction_length (int): Number of future time steps to estimate.
        mode (str): Mode of estimation ("timeseries_only", "text_only", "combined").

    Returns:
        str: Meta-prompt for ChatGPT.
    """
    prompt = (
        f"You are an AI assistant trained in data analysis and modeling. "
        f"Your task is to conduct a research-based timeseries estimation for the next {prediction_length} time steps "
        f"based on provided historical price movements and/or related news articles. "
        f"This analysis aims to explore patterns in the given dataset and should not be considered financial advice. "
        f"The input time series spans from {start_datetime} to {end_datetime}, with a granularity of {granularity}. "
        f"The estimation period extends from {end_datetime} to {pred_end_datetime}, maintaining the same granularity."
    )

    if mode == "timeseries_only":
        prompt += (
            "You will analyze the numerical patterns in historical prices and extrapolate potential movements. "
            f"The input prices are: {prices}. "
        )
    elif mode == "text_only":
        prompt += (
            "You will analyze sentiment and potential market impacts from the following news article content: "
            f"{text}. "
        )
    elif mode == "combined":
        prompt += (
            "You will use both historical price movements and relevant news sentiment analysis "
            f"to explore hypothetical market trends. The input prices are: {prices}. The news article states: {text}. "
        )
    else:
        raise ValueError(
            "Invalid mode. Choose from 'timeseries_only', 'text_only', or 'combined'."
        )

    prompt += (
        "\n\nPlease return your estimated values in a structured format as a  list of float numbers. "
        "Ensure the output follows this format strictly: "
        "\nPredicted Prices: value1, value2, ..., valueN. "
        f"The number of estimated values should be exactly {prediction_length}. "
    )

    return prompt

def finance_macd_metaprompt_generation(
    text: str,
    prices: List[float],
    start_datetime: str,
    end_datetime: str,
    pred_end_datetime: str,
    granularity: str,
    prediction_length: int,
    mode: str,
) -> str:

    prompt = (
        f"You are an AI assistant trained in data analysis and modeling. "
        f"Your task is to Predict the future Moving Average Convergence Divergence (MACD) values for the next {prediction_length} time steps "
        f"based on provided historical timeseries movements and/or related news articles. "
        # f"This analysis aims to explore patterns in the given dataset and should not be considered financial advice. "
        f"The input time series spans from {start_datetime} to {end_datetime}, with a granularity of {granularity}. "
        f"The estimation period extends from {end_datetime} to {pred_end_datetime}, maintaining the same granularity."
    )

    if mode == "timeseries_only":
        prompt += (
            "You will analyze the numerical patterns in historical prices. "
            f"The input prices are: {prices}. "
        )
    elif mode == "text_only":
        prompt += (
            "You will analyze sentiment and potential market impacts from the following news article content: "
            f"{text}. "
        )
    elif mode == "combined":
        prompt += (
            "You will use both historical price movements and relevant text sentiment analysis "
            f"The input prices are: {prices}. The news article states: {text}. "
        )
    else:
        raise ValueError(
            "Invalid mode. Choose from 'timeseries_only', 'text_only', or 'combined'."
        )

    prompt += (
        "\n\nPlease return your predicted MACD values in a structured format as a list of float numbers. Please predict the real possible values, do not use the naive linear extrapolation or similar methods"
        "Ensure the output follows this format strictly: "
        "\nPredicted Prices: value1, value2, ..., valueN. "
        f"The number of predicted values should be exactly {prediction_length}. "
    )

    return prompt

def finance_bb_metaprompt_generation(
    text: str,
    prices: List[float],
    start_datetime: str,
    end_datetime: str,
    pred_end_datetime: str,
    granularity: str,
    prediction_length: int,
    mode: str,
) -> str:

    prompt = (
        f"You are an AI assistant trained in data analysis and modeling. "
        f"Your task is to Predict the future upper Bollinger Band (BB) values  for the next {prediction_length} time steps "
        f"based on provided historical price movements and/or related news articles. "
        # f"This analysis aims to explore patterns in the given dataset and should not be considered financial advice. "
        f"The input time series spans from {start_datetime} to {end_datetime}, with a granularity of {granularity}. "
        f"The estimation period extends from {end_datetime} to {pred_end_datetime}, maintaining the same granularity."
    )

    if mode == "timeseries_only":
        prompt += (
            "You will analyze the numerical patterns in historical prices. "
            f"The input prices are: {prices}. "
        )
    elif mode == "text_only":
        prompt += (
            "You will analyze sentiment and potential market impacts from the following news article content: "
            f"{text}. "
        )
    elif mode == "combined":
        prompt += (
            "You will use both historical price movements and relevant news sentiment analysis "
            f"to explore hypothetical market trends. The input prices are: {prices}. The news article states: {text}. "
        )
    else:
        raise ValueError(
            "Invalid mode. Choose from 'timeseries_only', 'text_only', or 'combined'."
        )

    prompt += (
        "\n\nPlease return your estimated upper Bollinger Band (BB) values values in a structured format as a list of float numbers. "
        "Ensure the output follows this format strictly: "
        "\nPredicted Prices: value1, value2, ..., valueN. "
        f"The number of estimated values should be exactly {prediction_length}. "
    )

    return prompt

def parse_val_prediction_response(response: str) -> Union[List[float], None]:
    """
    Decodes the predicted prices from a response string.

    Args:
        response (str): The response containing the predicted prices.

    Returns:
        List[float]: A list of float numbers extracted from the response.
        None: If extraction fails.
    """
    match = re.search(r"Predicted Prices:\s*([-\d.,\s]+)", response)
    
    if match:
        try:
            price_list = [float(value) for value in match.group(1).split(',')]
            return price_list
        except ValueError:
            pass  # If conversion fails, try another approach
    
    # Alternative approach: Find all potential numbers in the response
    possible_numbers = re.findall(r"-?\d+\.\d+", response)
    if possible_numbers:
        try:
            return [float(num) for num in possible_numbers]
        except ValueError:
            pass  # If conversion fails, return None
    
    return None  # Return None if extraction fails

def finance_classification_metaprompt_generation(text=None, timestamps=None, prices=None, mode=None):
    time_series_data = ", ".join([f"{price}" for price in  prices])
    
    if mode == "combined": 
        meta_prompt = f"""
            You are a financial prediction expert with knowledge of advanced machine learning models and time-series analysis. 
            Your goal is to predict the stock trend (rise, neutral, or fall) based on the following inputs:

            1. **Time Series Stock Price Data**:
            - This data includes stock prices recorded at 1-hour intervals over the last month from {timestamps[0]} to {timestamps[-1]}.
            - Example data format:
                {time_series_data}

            2. **News Data**:
            - This includes news headlines and summaries relevant to the stock's company or sector.
            - Example data format:
                {text}

            ### Task:
            Analyze the provided time-series data and news to identify future trends of the stock performance. Ensure that the news data is used to supplement the insights from the time-series analysis, focusing on combining both inputs for a more accurate prediction.

            ### Output:
            Provide a prediction for the stock trend categorized one of the following labels:
            - "<-4%"
            - "-2% ~ -4%"
            - "-2% ~ +2%"
            - "+2% ~ +4%"
            - ">+4%"

            please think step-by-step and briefly explain how the combination of time-series data and news data led to the prediction;
            then wrap your final answer in the final predicted label in the format ^^^label^^^
        """
    
    elif mode == "text_only":
        meta_prompt = f"""
            You are a financial prediction expert with knowledge of advanced machine learning models and time-series analysis. 
            Your goal is to predict the stock trend with given labels based on the following input:

            **News Data**:
            - This includes news headlines and summaries relevant to the stock's company or sector.
            - Example data format:
                {text}
                
            ### Output:
            Provide a prediction for the stock trend categorized one of the following labels:
            - "<-4%"
            - "-2% ~ -4%"
            - "-2% ~ +2%"
            - "+2% ~ +4%"
            - ">+4%"

            ### Task:
            Analyze the news semantics to identify trends and patterns that could impact stock performance.
            Then wrap your final answer in the final predicted label in the format ^^^label^^^
        """
    
    elif mode == "timeseries_only":
        meta_prompt = f"""
            You are a financial prediction expert with knowledge of advanced machine learning models and time-series analysis. 
            Your goal is to predict the stock trend with given labels based on the following input:

            1. **Time Series Stock Price Data**:
            - This data includes stock prices recorded at 1-hour intervals over the last month from {timestamps[0]} to {timestamps[-1]}.
            - Example data format:
                {time_series_data}
            
            ### Output:
            Provide a prediction for the stock trend categorized one of the following labels:
            - "<-4%"
            - "-2% ~ -4%"
            - "-2% ~ +2%"
            - "+2% ~ +4%"
            - ">+4%"

            ### Task:
            Analyze the provided time-series data to identify trends and patterns that could impact stock performance. Focus solely on the time-series data for making predictions.
             then wrap your final answer in the final predicted label in the format ^^^label^^^
        """

    return meta_prompt

def parse_cls_response(answer):
    try:
        return  re.findall(r'\^\^\^(.*?)\^\^\^', answer)[-1]
    except:
        return  re.findall(r'\^+(.*?)\^+', answer)[-1]



def finance_correlation_metaprompt_generation(setting, sticker, time1, time2, in_price, news, time_news):
    
    time_interval = "1 hour" if setting == "long" else "5 minutes"
    
    if setting == "long":
        system_prompt ="You are an expert in finance and stock market analysis. Based on the given 30-day historical stock price time series and a financial analysis published at the last timestamp of the time series, your task is to predict the correlation between the stock's price fluctuations in the next 7 days and the analysis sentiment (positive correlation indicates that positive analysis leads to price increase and negative analysis leads to price decrease). Take into account external factors or market conditions that might affect stock price movement."
    else:
        system_prompt = "You are an expert in finance and stock market analysis. Based on the given 7-day historical stock price time series and a financial analysis published at the last timestamp of the time series, your task is to predict the correlation between the stock's price fluctuations in the next 1 day and the analysis sentiment (positive correlation indicates that positive analysis leads to price increase and negative analysis leads to price decrease). Take into account external factors or market conditions that might affect stock price movement."
    question = "Return your answer in one of the following without any other words: Strong Positive Correlation, Moderate Positive Correlation, No Correlation, Moderate Negative Correlation, Strong Negative Correlation."
    query = f"stock price of {sticker} between {time1} to {time2}, time interval is {time_interval}: \
            {in_price}\
            News published at {time_news}: \
            {news}\
            {question} Answer:"
    prompt = f"{system_prompt}\n\n{query}"
    
    return prompt




def finance_mcqa_metaprompt_generation(setting, sticker, time1, time2, in_price, news, time_news, question):
    time_interval = "1 hour" if setting == "long" else "5 minutes"
    if setting  == "long":
        system_prompt ="You are an expert in finance and stock market analysis. Your task is to answer the question based on the given 30-day historical stock price time series and a financial analysis published at the last timestamp of the time series. Return your answer only in the letter (A, B, C, or D). "
    else:
        system_prompt ="You are an expert in finance and stock market analysis. Your task is to answer the question based on the given 7-day historical stock price time series and a financial analysis published at the last timestamp of the time series. Return your answer only in the letter (A, B, C, or D). "
    query = f"stock price of {sticker} between {time1} to {time2}, time interval is {time_interval}: \
            {in_price}\
            News published at {time_news}: \
            {news}\
            Question: {question}. Give your answer in the letter (A, B, C, or D) without any other words. Answer:"
    prompt = f"{system_prompt}\n\n{query}"
    return prompt