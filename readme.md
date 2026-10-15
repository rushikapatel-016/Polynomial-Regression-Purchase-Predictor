# Polynomial Regression – Purchase Amount Predictor

## 1. Project Overview

The **Purchase Amount Predictor** is a Machine Learning project that uses Polynomial Regression to estimate a customer's purchase amount based on their review rating.

The project uses a retail sales dataset and Python libraries to train, evaluate, and save the model. A Flask web application allows users to enter a review rating and view the predicted purchase amount.

## 2. Project Objectives

- Understand Polynomial Regression.
- Load and preprocess a CSV dataset using Pandas.
- Train a Polynomial Regression model.
- Evaluate model performance using regression metrics.
- Save the trained model using Joblib.
- Develop a web interface using Flask, HTML, and CSS.

## 3. Technologies Used

- **Python** – Programming language
- **Pandas** – Data processing
- **NumPy** – Numerical calculations
- **Scikit-learn** – Machine Learning model and evaluation
- **Flask** – Web application framework
- **HTML** – Web page structure
- **CSS** – Web page styling
- **Joblib** – Saving and loading the trained model

## 4. Dataset

**Dataset filename:** `Fashion_Retail_Sales.csv`

The dataset contains retail sales information. The exact columns used by the model must match the columns available in the CSV file.

The intended model variables are:

- **Input Feature:** Review Rating
- **Target Variable:** Purchase Amount

Missing or invalid values in the selected columns should be handled before training.

## 5. Algorithm: Polynomial Regression

Polynomial Regression is a supervised Machine Learning technique used to model a nonlinear relationship between an input variable and a continuous output.

This project uses a polynomial of degree 2.

**Equation:**

y = b₀ + b₁x + b₂x²

Where:

- **x** = Customer review rating
- **y** = Predicted purchase amount
- **b₀** = Intercept
- **b₁, b₂** = Coefficients learned by the model

The model learns these coefficients from the training dataset and uses them to generate predictions.

## 6. Project Structure

```text
polynomial rregression project/
│
├── app.py
├── Fashion_Retail_Sales.csv
├── readme.md
└── requirements.txt
```

The trained model file and any additional files will be created only if required by the implementation.

## 7. Installation

Make sure Python is installed on your computer.

Open the terminal in the project folder and install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## 8. Model Training

The dataset is loaded and preprocessed before the Polynomial Regression model is trained.

The training process includes:

1. Loading the CSV dataset.
2. Selecting the input and target variables.
3. Handling missing or invalid values.
4. Splitting the dataset into training and testing sets.
5. Transforming the input features into polynomial features.
6. Training the regression model.
7. Evaluating model performance.
8. Saving the trained model for use in the web application.

## 9. Running the Web Application

After the model has been trained and saved, start the Flask application:

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

Enter a review rating within the permitted range and submit the form to view the predicted purchase amount.

**Note:** The application requires a compatible trained model. The training and application code must use the same input feature names and preprocessing steps.

## 10. Model Evaluation

The following metrics can be used to evaluate the model:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Indicates how well the model explains the variation in the target variable.

A model's actual performance must be determined from the evaluation results; it should not be assumed to be accurate without testing.

## 11. Limitations

- The model's performance depends on the quality and size of the dataset.
- Review rating alone may not be sufficient to predict purchase amount accurately.
- Predictions are estimates and may differ from actual purchase amounts.
- The model is intended for educational and demonstration purposes.

## 12. Future Scope

- Include additional relevant customer and sales features.
- Compare Linear Regression and Polynomial Regression.
- Experiment with different polynomial degrees.
- Improve model performance through feature selection and evaluation.
- Add data visualizations and prediction history.
- Improve the web interface and user experience.

## 13. Conclusion

This project demonstrates the basic Machine Learning workflow, including data preprocessing, Polynomial Regression, model evaluation, and integration with a Flask web application.

It provides a practical introduction to building and presenting a regression-based prediction system using Python.

