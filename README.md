
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
