# Iris Flower Classification

## Overview
This project uses Machine Learning to classify Iris flowers into three species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

based on sepal and petal measurements.

## Dataset
The Iris dataset contains:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Species

## Tools & Libraries
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Algorithm Used
Random Forest Classifier

## Project Workflow
1. Load Dataset
2. Data Cleaning
3. Feature Selection
4. Train-Test Split
5. Model Training
6. Prediction
7. Performance Evaluation
8. Data Visualization

## Results
The model successfully classifies Iris flower species with high accuracy.

Dataset shape: (150, 5)

First 5 rows:
    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0            5.1           3.5            1.4           0.2  Iris-setosa
1            4.9           3.0            1.4           0.2  Iris-setosa
2            4.7           3.2            1.3           0.2  Iris-setosa
3            4.6           3.1            1.5           0.2  Iris-setosa
4            5.0           3.6            1.4           0.2  Iris-setosa

Class distribution:
 Species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: count, dtype: int64

==================================================
MODEL COMPARISON
==================================================
Logistic Regression    | Test Accuracy:  93.33% | CV Mean:  95.83%
K-Nearest Neighbors    | Test Accuracy:  93.33% | CV Mean:  96.67%
Decision Tree          | Test Accuracy:  90.00% | CV Mean:  94.17%
Random Forest          | Test Accuracy:  90.00% | CV Mean:  95.00%
SVM (RBF Kernel)       | Test Accuracy:  96.67% | CV Mean:  96.67%

==================================================
BEST MODEL: SVM (RBF Kernel)
==================================================

Classification Report:
                  precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      0.90      0.95        10
 Iris-virginica       0.91      1.00      0.95        10

       accuracy                           0.97        30
      macro avg       0.97      0.97      0.97        30
   weighted avg       0.97      0.97      0.97        30

Confusion Matrix:
 [[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]



## Author
Bala Murugan

