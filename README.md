# Food Freshness Classification Model 

This project utilizes machine learning to classify whether fruits and vegetables are fresh or stale, supporting automated food-quality inspection and reducing waste.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 📖 About

Modern food logistics face two major problems:
  1. Unnecessary spoilage due to misjudging freshness
  2. Health risks caused by consuming spoiled produce

This project addresses those issues by building a machine-learning system that predicts: The type of produce, and whether the produce is fresh or stale

##  Features
1. Produce Type Classification (Random Forest)

A Random Forest classifier is used to identify which fruit or vegetable appears in an image (e.g., apple, banana, tomato).
Key features extracted:

  - Color histograms (hue, saturation, brightness)
  - Texture metrics using LBP & GLCM
  - Edge information to capture shape differences


2. Freshness Classification (Logistic Regression)

A Logistic Regression model classifies produce as fresh (0) or stale (1).

Features used to determine freshness:
  - Color intensity & brightness — dullness often indicates spoilage
  - Texture coarseness — rough/spotty surfaces indicate aging
  - Edge sharpness / wrinkle detection via Sobel filters
  - Surface irregularity captured by texture patterns

Positive coefficients corresponded to stale indicators, and negative coefficients corresponded to fresh

Getting Started

This section describes how to set up the environment, install dependencies, prepare the dataset, and run the classification pipeline.

# Prerequisites

Ensure the following are installed:

  - Python 3.8 or higher

  - pip (Python package manager)

  - Optional: a virtual environment tool such as venv or conda

## Install the required Python packages:

pip install scikit-learn opencv-python scikit-image numpy pandas matplotlib seaborn joblib

These packages support image processing, feature extraction, machine learning model training, and visualization.


# Installation

Clone the repository and install the project requirements:

git clone https://github.com/rgcurling/food-freshness-model.git
cd food-freshness-model
pip install -r requirements.txt


If your repository does not yet include a requirements.txt file, one can be generated based on the project dependencies.

# Dataset Setup

This project uses the "Fruits Fresh and Rotten" dataset from Kaggle:
https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification

After downloading the dataset, place the extracted image folders into:

data/images/


Each folder should correspond to a specific class label, such as:

fresh_apple/
rotten_apple/
fresh_banana/
rotten_banana/
...


The dataset includes categories such as oranges, bananas, apples, potatoes, okra, tomatoes, capsicum, cucumbers, and bitter gourd.


# Running the Pipeline

Once dependencies are installed and the dataset is prepared, run the full machine learning pipeline:
  - python src/index.py

This script performs the following steps:

  - Loads and preprocesses images
  - Extracts features (color histograms, texture metrics, and edge features)
  - Trains a Random Forest model for produce type classification
  - Trains a Logistic Regression model for freshness classification
  - Evaluates model performance

Saves the trained models using joblibENSE` for more information.
