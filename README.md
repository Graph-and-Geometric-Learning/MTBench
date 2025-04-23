<h1 align="center">
  <img src="assets/mtbench.png" width="200" /></a><br>
  <b>MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering</b><br>
</h1>

<div align="center" style="line-height: 1;">
  <a href=""><img alt="HuggingFace"
    src="https://img.shields.io/badge/Hugging_Face-MTBench-yellow?logo=huggingface"
    "/></a>
  <a href="https://arxiv.org/pdf/2503.16858"><img alt="Arxiv"
    src="https://img.shields.io/badge/arxiv-red?logo=arxiv"
    g"/></a>
  <a href=""><img alt="Website"
    src="https://img.shields.io/badge/Project-blue?logo=googlechrome"
    g"/></a>
</div>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [1. Abstract](#1-abstract)
- [2. Folder Structure](#2-folder-structure)
- [3. Dataset and Usage](#3-dataset-and-usage)
  - [Dependencies](#dependencies)
  - [Download Dataset](#download-dataset)
  - [Dataset Distribution](#dataset-distribution)
  - [Evaluation](#evaluation)
- [4. Baseline Results](#4-baseline-results)
  - [Results on Finance Data](#results-on-finance-data)
  - [Results on Weather Data](#results-on-weather-data)
- [5. Contribution and Future Work](#5-contribution-and-future-work)
- [6. Citation and License](#6-citation-and-license)

## 1. Abstract

We introduce **M**ultimodal **T**ime Series **Bench**mark (**MTBench**), a large-scale benchmark designed to evaluate large language models (LLMs) on time series and text understanding across financial and weather domains. MTBench comprises of paired time-series and textual data, including financial news with corresponding stock price movements and weather reports aligned with historical temperature records. Unlike existing benchmarks that focus on isolated modalities, MTBench provides a comprehensive testbed for models to jointly reason over structured numerical trends and unstructured textual narratives. The richness of MTBench enables formulation of diverse tasks that require a deep understanding of both text and time-series data, including time-series forecasting, semantic and technical trend analysis, and news-driven question answering (QA). These tasks target the model’s ability to capture temporal dependencies, extract key insights from textual context, and integrate cross-modal information. We evaluate state-of-the-art LLMs on MTBench analyzing their effectiveness in modeling the complex relationships between news narratives and temporal patterns. Our findings reveal significant challenges in current models, including difficulties in capturing long-term dependencies, interpreting causality in financial and weather trends, and effectively fusing multimodal information.

## 2. Folder Structure

```
MTBench/
│── data/                           # Downloaded datasets
    ├── raw/                        # Text or timeseries only dataset
    ├── processed/                  # Task-specific dataset
│── data_preparation/               # Dataset preparation scripts
    ├── weather/                    # Scripts for weather data processing
    ├── finance/                    # Scripts for financial data processing
│── evaluation/                     # Evaluation scripts for benchmarking
    ├── weather/                    # Evaluation scripts for weather data
    ├── finance/                    # Evaluation scripts for finance data
    |── api_call.py                 # Functions for calling LLM APIs
│── requirements.txt                # Dependencies
|── download_raw_dataset.py         # Download the raw dataset
|── download_processed_dataset.py   # Download all processed dataset
│── README.md                       # Project documentation
```

## 3. Dataset and Usage

MTBench introduces cross-domain dataset covering two domains: **weather** and **finance**. These datasets are designed to evaluate large language models (LLMs) on temporal reasoning and question-answering tasks. Each dataset consists of structured time series data and textual questions that require understanding of time-dependent trends.

### Dependencies

Run the following commands to create a conda environment for MTBench

```bash
git clone https://github.com/Graph-and-Geometric-Learning/MTBench.git
cd MTBench

conda create -n MTBench python=3.10.14
source activate MTBench
pip install -r requirements.txt
```

### Download Dataset

Download the raw dataset by running `python download_raw_dataset.py`. We provide the scripts to preprocess the raw data in `data_prepraration`. Scripts will:

* generate trend labels and calcuate technical indicators
* generate multi-choice QA samples and correlation labels

For your convenience, you can download all the processed data by

```bash
python download_processed_dataset.py
```

### Dataset Distribution

Distributions of financial news impact duration and financial news categories:

<div align="center">
  <img src="assets/finance_duration.png" alt="Finance Duration Distribution" width="48%"/>
  <img src="assets/finance_type.png" alt="Finance Report Type Distribution" width="48%"/>

</div>
Distributions of severe weather duration and their types:
<div align="center" style="display: flex; justify-content: space-between;">
  <img src="assets/weather_duration.png" alt="Weather Duration Distribution" width="48%"/>
  <img src="assets/weather_event.png" alt="Weather Event Distribution" width="48%"/>
</div>

### Evaluation

To evaluate models on MTBench, you need to:

1. Set up API keys for LLMs in `evaluation/api_call.py`
2. Choose the domain, evaluation task and the setting
3. Run the corresponding evaluation script

For example, to evaluate time series trend classification on financial data, you need to set the arguments in `evaluation/finance/run_trend_classification.sh` :

```
API_NAME="gpt-4o"  # choose the LLM to be evaluated
MODE="combined"    # choose the input type, select from ["timeseries_only", "combined"]
IN_DAYS=30         # length of input time series
OUT_DAYS=7         # length of output time series

python trend_classification.py \
    --dataset_folder="../../data/processed/finance/aligned_in${IN_DAYS}days_out${OUT_DAYS}days" \
    --save_path="../../results/finance/trend_classification_in${IN_DAYS}_out${OUT_DAYS}/${API_NAME}_${MODE}" \
    --model=$API_NAME \
    --mode=$MODE

```

Then run the evaluation script:

```bash
  cd evaluation/finance
  bash run_trend_classification.sh
```

Results are saved to `results/finance/trend_classification` correspondingly.

## 4. Baseline Results

We benchmark several state-of-the-art LLMs. The performance varies across different temporal reasoning tasks, highlighting areas for improvement in existing LLMs.

### Results on Finance Data

Evaluation on short-term finance data (*e.g.,* 7-day input, 1-day output). "➡️" indicates the performance change between *Time Series-Only* and *Time Series + Text* Input.


|              |  Trend Prediction<br/> (ACC)  | Technical Indicator (MSE) | Correlation (ACC) | MCQA (ACC) |
| ------------ | :---------------------------: | :-----------------------: | ----------------- | ---------- |
| **GPT-4o**   | 40.93 ➡️ 42.81 | 0.430 ➡️ 0.365 | 53.6              | 65.1       |
| **Gemini**   |   41.30 ➡️ 47.30   | 0.482 ➡️ 0.384 | 51.8              | 63.6       |
| **Claude**   |   41.20 ➡️ 44.90   | 0.241 ➡️ 0.373 | 50.4              | 75.6       |
| **DeepSeek** |   40.53 ➡️ 45.12   | 0.435 ➡️ 0.352 | 50.0              | 77.6       |

### Results on Weather Data


|              | Temperature Forecasting (MSE) |  Trend Prediction (ACC)  | Temperature Difference (MSE) | MCQA (ACC) |
| ------------ | :---------------------------: | :-----------------------: | :--------------------------: | :--------: |
| **GPT-4o**   |   21.67 ➡️ 17.55   | 23.07 ➡️ 43.54 |  27.06 ➡️ 18.84  |    41.7    |
| **Gemini**   |   25.75 ➡️ 24.31   | 17.91 ➡️ 51.76 |  35.72 ➡️ 23.21  |    43.4    |
| **Claude**   |   30.34 ➡️ 22.48   | 33.23 ➡️ 56.87 |  21.03 ➡️ 19.10  |    51.8    |
| **DeepSeek** |   31.02 ➡️ 29.38   | 16.89 ➡️ 25.17 |  49.28 ➡️ 44.99  |    46.7    |

## 5. Contribution and Future Work

We invite contributions to improve MTBench, including:

* Expanding dataset diversity with new domains.
* Enhancing task formulation for more complex temporal reasoning.
* Developing evaluation metrics tailored for multimodal time series reasoning.
* Designing novel and effective architectures and altorihtms for multimodal time series reasoning.

## 6. Citation and License

This code repository is licensed under [the MIT License](LICENSE-CODE).

If you find MTBench useful, please consider citing our paper:

```bibtex
@article{chen2025mtbench,
  title={MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering},
  author={Chen, Jialin and Feng, Aosong and Zhao, Ziyu and Garza, Juan and Nurbek, Gaukhar and Qin, Cheng and Maatouk, Ali and Tassiulas, Leandros and Gao, Yifeng and Ying, Rex},
  journal={arXiv preprint arXiv:2503.16858},
  year={2025}
}
```
