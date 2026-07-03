# Hate Speech Detection using Machine Learning

## Project Overview
This project is a Machine Learning based web application that detects whether a given text contains hate speech or normal speech.

The system uses Natural Language Processing (NLP) techniques and a Logistic Regression model to classify text into hate speech or non-hate speech categories.

---

## Features
- Detects hate speech from text input
- Web-based user interface
- Machine Learning text classification
- Real-time prediction
- Simple and user-friendly design

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- TF-IDF Vectorizer
- Logistic Regression
- Visual Studio Code

---

## Project Structure

```text
hate-speech-detection/
│
├── data/
│   └── hate_speech.csv
│
├── models/
│   └── model.pkl
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
```
---

## Dataset
The dataset used for this project was downloaded from Kaggle.

It contains text samples labeled as:
- Hate Speech
- Offensive Language
- Normal Speech

---

## Model Training Process
1. Data collection
2. Data preprocessing
3. Text vectorization using TF-IDF
4. Model training using Logistic Regression
5. Model evaluation
6. Model saving

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt