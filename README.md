
# Fake News Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-success)]()

## Table of Contents
- [Introduction](#introduction)
- [Why Does Fake News Exist?](#why-does-fake-news-exist)
- [How to Recognize Fake News?](#how-to-recognize-fake-news)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Usage](#usage)
- [Models and Results](#models-and-results)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

In today's digital world, fake news has become a significant challenge. With the rapid spread of information on social media and online platforms, distinguishing between true and false information is becoming increasingly difficult.

This project aims to tackle this problem using **Machine Learning** and **Natural Language Processing (NLP)** techniques to automatically classify news articles as real or fake.

## Why Does Fake News Exist?

There are several reasons for the creation of fake news:

- **Manipulation and disinformation**: Created to influence public opinion (e.g., during elections or wars)
- **Clickbait**: Sensational false headlines attract more views, leading to higher ad revenue
- **Errors**: Sometimes false news results from mistakes or lack of verification rather than intentional deception

## How to Recognize Fake News?

Some key strategies include:

- **Check the source**: Is it reputable and verified?
- **Cross-verify**: Reliable news will appear in multiple trusted outlets
- **Examine language**: Overly emotional, sensational, or grammatically incorrect content may be suspicious
- **Look at the publication date**: Old or out-of-context news can be misleading

## Project Overview

Our goal is to build a **machine learning model** that classifies news articles into two categories:
- **True information (0)**
- **Fake news (1)**

### Methodology

1. **Data Cleaning**: Fix CSV formatting issues and remove inconsistencies
2. **Data Preprocessing**: Remove nulls, duplicates, and irrelevant columns
3. **Text Processing**: Lowercase, tokenize, remove stopwords, and lemmatize
4. **Feature Engineering**: Generate Word2Vec embeddings (100-dimensional vectors)
5. **Model Training**: Train and compare Logistic Regression, Random Forest, and SVM
6. **Evaluation**: Measure performance using accuracy, precision, recall, and F1-score

## Dataset

The dataset contains **10,321 news articles** after cleaning, with the following columns:

| Column | Description |
|--------|-------------|
| `title` | Article title |
| `text` | Article content |
| `subject` | Article category/topic (News, Politics, World News, etc.) |
| `isfake` | Target label (0 = true, 1 = fake) |
| `title_content` | Combined title and text |
| `processed` | Preprocessed text (lowercased, cleaned, stopwords removed, lemmatized) |

### Subject Distribution

The dataset includes 8 different news categories:
- **politicsNews** (2,920 articles) - 28.3%
- **worldnews** (2,691 articles) - 26.1%
- **News** (1,931 articles) - 18.7%
- **politics** (1,270 articles) - 12.3%
- **leftnews** (882 articles) - 8.5%
- **Government News** (303 articles) - 2.9%
- **USNews** (165 articles) - 1.6%
- **Middleeast** (159 articles) - 1.5%

**Political news articles**: 4,190 (40.6% of total dataset)

## Project Structure

```

fake-news-detection/
│
├── data/
│   ├── news_dataset.csv                    # Original raw dataset
│   ├── news_dataset1.csv                   # After removing trailing semicolons
│   ├── news_dataset_semicolon.csv          # After header separator replacement
│   ├── news_dataset_no_quotes.csv          # After removing quotes
│   ├── news_dataset_clean.csv              # After cleaning special characters
│   ├── news_dataset_clean2.csv             # Final cleaned CSV (no nulls/duplicates)
│   └── news_dataset_preprocessed.json      # Preprocessed with 'processed' column
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb              # Data cleaning pipeline
│   ├── 02_preprocessing.ipynb              # Text preprocessing and NLP
│   ├── 03_word2vec_training.ipynb          # Word2Vec embeddings
│   └── 04_model_training.ipynb             # ML model training and evaluation
│
├── src/
│   ├── init.py
│   ├── data_cleaning.py                    # CSV cleaning functions
│   ├── preprocessing.py                    # Text preprocessing functions
│   ├── feature_extraction.py               # Word2Vec implementation
│   └── models.py                           # ML model definitions
│
├── reports/
│   ├── profile_report.html                 # YData profiling report
│   └── model_results.txt                   # Classification reports
│
├── requirements.txt                         # Python dependencies
├── README.md                               # Project documentation
├── .gitignore                              # Git ignore file
└── LICENSE                                 # MIT License

```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```
2. **Create a virtual environment (recommended)**:
```bash
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install -r requirements.txt

```

4. **Download NLTK data**:
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```
5. **requirements.txt**
```
textpandas>=1.3.0
numpy>=1.21.0
nltk>=3.6.0
gensim>=4.0.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
ydata-profiling>=4.0.0
```
## Pipeline Workflow

Phase 1: Data Cleaning
The raw CSV file had multiple formatting issues that required systematic cleaning:

Remove trailing semicolons from lines
Replace commas with semicolons in header
Remove quotes at the beginning and end of lines
Replace double quotes with semicolons and remove special characters
Load cleaned data with proper separator

Phase 2: Data Preprocessing

Remove null values from all columns
Drop duplicate rows (20 duplicates found)
Remove unnecessary columns (date column)
Reset index for clean DataFrame

Phase 3: Text Preprocessing
Comprehensive NLP preprocessing pipeline

Phase 4: Feature Extraction (Word2Vec)
Convert preprocessed text into 100-dimensional vectors:

Phase 5: Model Training
Train three different classifiers and compare results:

## Models and Results

### Model Performance Comparison

| Model | Accuracy | Precision (True) | Recall (True) | F1-Score (True) | Precision (Fake) | Recall (Fake) | F1-Score (Fake) |
|-------|----------|------------------|---------------|-----------------|------------------|---------------|-----------------|
| **Logistic Regression** | **97%** | 0.97 | 0.97 | **0.97** | 0.96 | 0.96 | **0.96** |
| **Support Vector Machine** | **97%** | 0.97 | 0.97 | **0.97** | 0.97 | 0.97 | **0.97** |
| **Random Forest** | 95% | 0.95 | 0.96 | 0.96 | 0.95 | 0.94 | 0.95 |


### Detailed Results

#### Logistic Regression

**Key Strengths:**
- Excellent balance between precision and recall
- Fast training and prediction time
- Low computational requirements
- Good interpretability with coefficient weights

#### Support Vector Machine (SVM)

**Key Strengths:**
- Best overall performance with perfect balance
- Excellent generalization
- Robust to high-dimensional data
- Effective with clear margin of separation

#### Random Forest


**Key Strengths:**
- Good performance with feature importance insights
- More interpretable model
- Robust to outliers and noise
- No hyperparameter tuning required

## Conclusion

### Key Observations

1. **Model Performance**
   - Logistic Regression and SVM achieved the highest accuracy at **97%**
   - Random Forest performed well at **95%** accuracy
   - All models demonstrate high effectiveness in classifying fake vs. true news
   - The minimal performance gap (2%) suggests the feature engineering was strong

2. **Feature Engineering Success**
   - Text preprocessing (lowercasing, tokenization, stopword removal, lemmatization) significantly improved model performance
   - **Word2Vec embeddings** effectively captured semantic meaning in 100-dimensional space
   - The average word vector approach for document representation proved highly effective

3. **Pipeline Effectiveness**
   - The complete pipeline (data cleaning → text preprocessing → Word2Vec → ML classifiers) proved highly effective
   - Achieving **97% accuracy** demonstrates viability for real-world applications
   - The systematic data cleaning approach resolved CSV formatting issues successfully
  
## Recommendations
  
   **For Production Deployment:**
- **Primary Choice**: Logistic Regression or SVM (97% accuracy, balanced performance)
- **Benefits**: Fast inference, low computational cost, high accuracy
- **Use Case**: Real-time fake news detection in news feeds or social media

**For Research/Analysis:**
- **Primary Choice**: Random Forest (95% accuracy with interpretability)
- **Benefits**: Feature importance analysis, understanding which words/patterns indicate fake news
- **Use Case**: Understanding fake news characteristics and patterns

  ## Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing and array operations

### NLP & Text Processing
- **NLTK** - Natural language toolkit for tokenization, stopwords, lemmatization
- **Gensim** - Word2Vec implementation for word embeddings
- **Regular Expressions (re)** - Text cleaning and pattern matching

### Machine Learning
- **Scikit-learn** - ML models and evaluation metrics
  - LogisticRegression
  - RandomForestClassifier
  - Support Vector Machine (SVC)
  - train_test_split
  - classification_report

### Data Analysis & Visualization
- **YData Profiling** - Automated exploratory data analysis
- **Jupyter Notebook** - Interactive development environment

-  Dataset source: News articles from various sources (2016-2017)
- Inspired by the growing need to combat misinformation in digital media
- Thanks to the open-source community for amazing tools (NLTK, Gensim, Scikit-learn)
- Special thanks to all contributors and supporters

## Contact

**Project Maintainer**: Milica Antic

- 💼 LinkedIn: https://www.linkedin.com/in/milica-antic-ds/

