# Fake News Detection Using Machine Learning

## Table of Contents
- [Introduction](#introduction)
- [Why Does Fake News Exist?](#why-does-fake-news-exist)
- [How to Recognize Fake News?](#how-to-recognize-fake-news)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Results](#models-and-results)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

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

1. **Data Preprocessing**: Clean, normalize, and prepare the text data
2. **Text Vectorization**: Convert text into numerical vectors using **Word2Vec** embeddings
3. **Model Building**: Train multiple classification models
4. **Evaluation**: Compare model performance to determine the most effective algorithm

## Dataset

The dataset contains **10,321 news articles** with the following columns:

| Column | Description |
|--------|-------------|
| `title` | Article title |
| `text` | Article content |
| `subject` | Article category/topic (News, Politics, World News, etc.) |
| `date` | Publication date |
| `isfake` | Target label (0 = true, 1 = fake) |
| `title_content` | Combined title and text |
| `processed` | Preprocessed text (lowercased, cleaned, stopwords removed, lemmatized) |

### Subject Distribution

The dataset includes 8 different news categories:
- politicsNews (2,920 articles)
- worldnews (2,691 articles)
- News (1,931 articles)
- politics (1,270 articles)
- leftnews (882 articles)
- Government News (303 articles)
- USNews (165 articles)
- Middleeast (159 articles)

## Project Structure
