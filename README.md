# Online Shoppers Purchasing Intention

This project analyzes the **Online Shoppers Purchasing Intention** dataset from the UCI Machine Learning Repository. The dataset contains information about visitors to an online shopping website and aims to predict whether a visitor will make a purchase based on various features.

## Dataset

- **Source**: UCI Machine Learning Repository
- **Link**: [Online Shoppers Purchasing Intention Data Set](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

## Technologies

This project was created using Python in a Jupyter Notebook. The primary libraries used include:

- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models and tools

## Features

The dataset includes the following features:
- **Administrative**: Number of pages visited related to website administration
- **Informational**: Number of pages visited related to website information
- **ProductRelated**: Number of pages visited related to products
- **BounceRates**: Percentage of visitors who leave after viewing only one page
- **ExitRates**: Percentage of visitors who leave the site from a particular page
- **PageValues**: Average value of a page visited
- **SpecialDay**: Closeness of the visit to a special day
- **OperatingSystems, Browser, Region, TrafficType**: Technical information about the visitor
- **VisitorType**: Whether the visitor is a returning or new visitor
- **Weekend**: Whether the visit was on a weekend
- **Revenue**: Whether the visit resulted in a purchase (target variable)

## Objective

The primary objective of this project is to build a predictive model to identify whether a visitor will make a purchase, based on the given features. The project involves data preprocessing, exploratory data analysis, feature selection, and the development of various machine learning models.

### Results

The following machine learning models were used to predict purchasing intention on the unbalanced dataset:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**

#### Initial Model Accuracy

- **Logistic Regression**: 87.10%
- **Decision Tree**: 85.69%
- **Random Forest**: 88.77%
- **Gradient Boosting**: 88.77%

#### Model Evaluation and Improvement

Hyperparameter tuning was performed, and cross-validation was used to assess model performance:

- **Logistic Regression**: Best cross-validation accuracy: 88.57%
- **Decision Tree**: Best cross-validation accuracy: 88.69%
- **Random Forest**: Best cross-validation accuracy: 90.13%
- **Gradient Boosting**: Best cross-validation accuracy: 90.19%

#### Accuracy After Hyperparameter Tuning

- **Logistic Regression**: 87.06%
- **Decision Tree**: 87.88%
- **Random Forest**: 88.93%
- **Gradient Boosting**: 89.09%

Overall, the Gradient Boosting model achieved the highest accuracy after hyperparameter tuning, followed closely by the Random Forest model.

### Balancing the Dataset

The dataset was balanced using SMOTE with Tomek Links, with the sampling strategy set to majority. This improved the model accuracies as follows:

- **Logistic Regression**: 89.85%
- **Decision Tree**: 90.96%
- **Random Forest**: 93.58%
- **Gradient Boosting**: 93.58%

### Summary

Balancing the dataset using SMOTE with Tomek Links led to significant improvements in model accuracy. The Gradient Boosting and Random Forest models achieved the highest accuracy at 93.58% after balancing the dataset. Therefore, these models are the most suited for predicting the customer’s purchase intention.

## Conclusion

This project analyzed the Online Shoppers Purchasing Intention dataset and developed machine learning models to predict whether a visitor will make a purchase. The models achieved high accuracy after balancing the dataset using SMOTE with Tomek Links. The Gradient Boosting and Random Forest models were the most accurate, achieving an accuracy of 93.58%. This project demonstrates the importance of data preprocessing and model selection in improving predictive performance.
