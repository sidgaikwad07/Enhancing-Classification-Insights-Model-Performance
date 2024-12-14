# Enhancing-Classification-Insights-Model-Performance
This repository contains code for a data mining task which is used for the multi-class classification data. The model is a Voting Classifier that combines the predictions of a Decision Tree Classifier, a Random Forest Classifier, and a K-Nearest Neighbors Classifier. The code also includes data preprocessing, outlier detection, and saving and loading the trained model. The goal is to provide a reliable way to make predictions on new data.

# Table of Contents
## 1. Project Structure
## 2. Coding Environment
## 3. Code Description
## 4. Instructions

## 1. Project Structure
### 1.1) Main Code: The main code is provided in a Python script named model_deploy.py.
### 1.2) Data Files: The code assumes the presence of specific data files:
      1) train.csv : The training dataset.
      2) add_train.csv : Additional training data.
      3) test.csv : The test dataset.
### 1.3) Model Saving: The trained machine learning model is saved as a .pkl file, named VotingClassifier.pkl.
### 1.4) Predictions: Predictions made on the test data are saved as a CSV file named s4771984.csv.

## 2. Coding Environment
### 2.1) Operating System: Mac OS
### 2.2) Python Version: 3.11
### 2.3) Additional Packages Installed: scikit-learn, pandas, numpy, joblib

### 3. Code Description
This Python code includes the implementation of a machine learning model for classification tasks using the scikit-learn library. The code performs the following tasks:
### 3.1) Data Preprocessing: The code reads training data and additional training data from CSV files, preprocesses the data by filling missing values, and detects and removes outliers.
### 3.2) Model Training: It trains a voting classifier that combines three different classifiers: DecisionTreeClassifier, RandomForestClassifier, and KNeighborsClassifier. The code saves the trained model as a pickle file.
### 3.3) Model Prediction: The code reads test data, preprocesses it, and uses the trained model to make predictions. It saves the predictions to a CSV file.
### 3.4) Output Formatting: The code formats the model's predictions, calculates accuracy and F1 score, and writes the results to a CSV file named "s4771984.csv".

## 4. Instructions
### 4.1) Environment Setup: Ensure you have Python 3.11 or a compatible version installed on your system. Install the required packages using the following command:
      1) pip install scikit-learn
      2) pip install pandas
      3) pip install numpy
      4) pip install joblib
### 4.2) Run the Code: Run the code using a Python IDE or command-line interface. The code reads training data from "train.csv" and additional training data from "add_train.csv". Ensure these files are in the same directory as the code file.
### 4.3) Data Pre-Processing: Data preprocessing is an essential step to prepare the data for training and prediction. The following preprocessing steps are implemented in the code:
      1) Handling missing values: Missing values in numeric columns are filled with the median, and missing values in categorical columns are filled with the mode.
      2) Outlier detection: An Isolation Forest is used to detect and remove outliers from the training data.
      3) The code uses these preprocessing techniques to ensure the data is clean and suitable for training the machine learning model.
      4) The code will preprocess the training data, detect and remove outliers, and train a voting classifier model. The trained model will be saved as "VotingClassifier.pkl" in the     current directory.
### 4.4) Model Prediction: The code reads test data from "test.csv," preprocesses it, and uses the trained model to make predictions.
### 4.5) Output File: The predictions, accuracy, and F1 score will be saved in a CSV file named "s4771984.csv" in the current directory.
