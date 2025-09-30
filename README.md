# Fake_News_Detection_using_Ensemble_Machine__Learning

A robust machine learning project for detecting fake news articles using ensemble models and advanced text feature engineering. This repository demonstrates a complete pipeline from data preprocessing to model evaluation, with extensive analysis and performance metrics.

# Project Overview
This project predicts whether a given news article is real or fake using classic and ensemble machine learning classifiers. The workflow applies natural language processing techniques, handcrafted text feature extraction, and multiple classification models evaluated for accuracy and robustness.

# Key Features
Data cleaning and preprocessing, including stopword removal and normalization.

Extraction of statistical features (word count, character count, etc.) from news content.

TF-IDF vectorization with unigrams and bigrams for feature representation.

Feature selection using chi-squared statistical testing, keeping top 8,000 features for speed.

Implementation of baseline ML models: Logistic Regression, Random Forest, Naive Bayes, Decision Tree.

Ensemble model using soft Voting Classifier for optimal performance.

Evaluation with accuracy, ROC AUC, and confusion matrix visualizations.

# Dataset
The dataset consists of news articles labeled as fake (0) or real (1).

Loaded from FakeDataset.csv and cleaned for missing or inconsistent labels.

# Installation
cd fake-news-detection
pip install -r requirements.txt

# Usage
# Prepare the dataset:

Place your labeled CSV news dataset as content/WELFakeDataset.csv in the root directory.

# Run the notebook or main script:

Open and execute FakeNewsPrediction.ipynb or

Run the main Python script for training and evaluation.

# Model Training & Evaluation
Data is split into training and test sets, with stratification for label balance.

Key models and their test accuracies:

Logistic Regression: 92.6%

Random Forest: 94.2%

Naive Bayes: 87.9%

Decision Tree: 91.6%

Ensemble Voting Classifier:

Accuracy: 92.3%

ROC AUC: 0.922

Cross-validated ROC AUC: 0.97 (mean).

# Results Visualization
Confusion matrix and classification reports are generated for all models.

Feature importances and most common terms in fake/real news are plotted for interpretability.

WordClouds for visualizing key terms in fake and real news articles.

# Project Structure
FakeNewsPrediction.ipynb - Main notebook with code and results.

content/WELFakeDataset.csv - (Dataset, not included)

requirements.txt - Python dependencies.

Supporting scripts for data cleaning, feature extraction, and model wrappers.

# Acknowledgments
Built using Scikit-learn, Pandas, NLTK, and Matplotlib.
