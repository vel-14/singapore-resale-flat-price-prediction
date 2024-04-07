# singapore-resale-flat-price-prediction

This project aims to predict resale flat prices in Singapore using machine learning techniques, specifically Random Forest Regression. The model is trained on historical transaction data of resale flats in Singapore and is capable of providing accurate price predictions based on various factors such as location, flat type, floor area, lease remaining, and amenities.

## Table of Contents

- Data Collection
- Data Cleansing and Preprocessing
- Exploratory Data Analysis (EDA)
- Model Selection
- Model Training
- Making Predictions
- Conclusion
- Data Collection

The first step in building the model is collecting historical transaction data of resale flats in Singapore. The data can be obtained from government websites. Ensure that the data includes relevant features such as flat type, location, floor area, lease remaining, and resale price.

## Data Cleansing and Preprocessing

Once the data is collected, it needs to be cleansed and preprocessed to ensure its quality and suitability for training the model. This involves handling missing values, removing duplicates, encoding categorical variables, scaling numerical features, and performing any necessary transformations.

## Exploratory Data Analysis (EDA)

EDA is performed to gain insights into the data and understand the relationships between different features and the target variable (resale price). Visualizations such as scatter plots, histograms, and correlation matrices are used to identify patterns and trends in the data.

## Model Selection

Several machine learning models are evaluated to determine the best-performing model for predicting resale flat prices. Models such as Linear Regression, Decision Trees and Random Forest are considered. The model with the highest performance metrics, such as R-squared and Mean Absolute Error (MAE), is selected for further training.

## Model Training

The selected Random Forest Regression model is trained on the preprocessed data. The training dataset is split into training and validation sets to evaluate the model's performance.

## Making Predictions

Once the model is trained, it can be used to make predictions on new data. Users can input the features of a resale flat, such as flat type, location, floor area, and lease remaining, and the model will output a predicted resale price. The model's predictions can help buyers and sellers make informed decisions in the real estate market.

## Conclusion

In conclusion, the Random Forest ML model developed in this project provides a powerful tool for predicting resale flat prices in Singapore. By leveraging historical transaction data and advanced machine learning techniques, the model offers accurate price predictions, aiding buyers, sellers, and real estate professionals in making informed decisions in the Singapore housing market.
