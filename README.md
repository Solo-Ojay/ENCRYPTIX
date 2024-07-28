# Encryptix Machine Learning Internship Tasks

This repository contains code and documentation for three distinct machine learning tasks:

1. **Credit Card Fraud Detection**
2. **Customer Churn Prediction**
3. **Spam SMS Detection**

## Task 1: Credit Card Fraud Detection

### Overview
The Credit Card Fraud Detection project focuses on identifying fraudulent transactions in credit card data. It involves preprocessing, feature engineering, model training, and performance evaluation.

### Tasks
1. **Data Preprocessing**:
   - Load and explore the credit card dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
   - Handle missing values and encode categorical variables.
   - Balance the dataset using upsampling.

2. **Feature Engineering**:
   - Create features from categorical data using `LabelEncoder`.
   - Drop unnecessary columns and split data into features and target variables.

3. **Model Training and Evaluation**:
   - Train a `RandomForestClassifier` on the training data.
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, ROC AUC, and PR AUC.
   - Visualize confusion matrix, ROC curve, and precision-recall curve.

### Files
- [credit_card_fraud_detection.ipynb](Encryptix Task 4.ipynb): Jupyter notebook containing code and analysis for credit card fraud detection.

## Task 2: Customer Churn Prediction

### Overview
The Customer Churn Prediction project is aimed at predicting whether a customer will leave a service. This project involves data preprocessing, feature engineering, and model training.

### Tasks
1. **Data Preprocessing**:
   - Load and explore the customer churn dataset from [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction).
   - Handle missing values and drop unnecessary columns.
   - Encode categorical features using one-hot encoding.

2. **Feature Scaling and Balancing**:
   - Scale numerical features using `MinMaxScaler`.
   - Use SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.

3. **Model Training and Evaluation**:
   - Train a `RandomForestClassifier` on the resampled training data.
   - Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

### Files
- `customer_churn_prediction.ipynb`: Jupyter notebook containing code and analysis for customer churn prediction.
  
## Task 3: Spam Detection

### Overview
The Spam Detection project aims to classify SMS messages as either "ham" (legitimate) or "spam". This involves preprocessing text data, training a machine learning model, and making predictions on new messages.

### Tasks
1. **Data Preprocessing**:
   - Load and clean the SMS dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
   - Combine and drop unnecessary columns.
   - Convert labels to binary format (ham: 0, spam: 1).

2. **Text Vectorization**:
   - Use `CountVectorizer` to convert text data into numerical feature vectors.

3. **Model Training and Evaluation**:
   - Train a `MultinomialNB` (Naive Bayes) model on the training data.
   - Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Make predictions and classify individual SMS messages.

4. **Interactive Prediction**:
   - Implement a simple user interface to input SMS text and predict if it's ham or spam.

### Files
- `Encryptix Task 4.ipynb`: Jupyter notebook containing code and analysis for spam detection.

## Installation

To run these notebooks, you will need Python and the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Spam Detection**: Open `spam_detection.ipynb` in Jupyter Notebook, follow the steps for preprocessing, training, and predicting spam messages.
2. **Credit Card Fraud Detection**: Open `credit_card_fraud_detection.ipynb`, and follow the steps for data processing, model training, and evaluation.
3. **Customer Churn Prediction**: Open `customer_churn_prediction.ipynb`, and execute the code for preprocessing, feature engineering, and model evaluation.
