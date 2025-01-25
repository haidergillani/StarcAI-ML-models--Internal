# Internal---STARC-AI-Sentiment-Modification-for-Financial-Communications
STARC is an advanced text manipulation model that aims to revolutionize how companies interact with 
shareholders and investors via automated sentiment modification. Leveraging advanced NLP, STARC allows 
businesses to strategically improve investor confidence, attract capital, and favorably influence stock 
pricing even in challenging financial times

This repository contains machine learning models and utilities for STARC AI's sentiment analysis and financial language processing.

## Project Structure

```
StarcAI-ML-models--Internal/
├── models/                  # Core model implementations
│   ├── FLS_Model.py        # Financial Language Sentiment model
│   ├── Tone_Model.py       # Tone analysis model
│   └── ...
├── benchmarks/             # Benchmarking tools and scripts
│   ├── setup_benchmark.py
│   └── ...
├── cloud/                  # Cloud deployment and functions
│   ├── STARC_Cloud_Request.py
│   └── ...
├── training/              # Model training scripts
│   └── Training_Web_Sentiment_Analysis.py
├── tests/                 # Test files
│   └── EXTRA_Sentiment_Test.py
├── utils/                 # Utility functions and helpers
│   ├── Web_Scrapping_Text_Cleaning.py
│   ├── your_tokenizer.py
│   └── ...
└── data/                  # Data handling and datasets
    └── Dataset.py
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For ONNX-specific requirements:
```bash
pip install -r requirements_onnx.txt
```

## Usage

See individual module documentation for specific usage instructions.
