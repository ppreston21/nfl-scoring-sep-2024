import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import argparse
import os


# Function to load CSV file
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit(1)


# Function to define and prepare the features and target variable
def prepare_data(df):
    X = df[['total_snaps', 'yards_gained']]  # Adjust feature names to match your CSV
    y = df['total_points']  # Adjust target variable
    return X, y


# Function to run Linear and Quadratic regression
def run_models(X, y):
    # Prepare polynomial features (quadratic)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Linear regression
    model_linear = LinearRegression()
    model_linear.fit(X, y)
    y_pred_linear = model_linear.predict(X)
    r2_linear = r2_score(y, y_pred_linear)

    # Quadratic regression
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)

    return model_linear, model_poly, r2_linear, r2_poly, poly


# Function to predict PTS using linear regression
def predict_pts_linear(model, plays, yds):
    """Predicts PTS using linear regression."""
    intercept = model.intercept_
    coef = model.coef_
    pts = intercept + coef[0] * plays + coef[1] * yds
    return pts


# Function to predict PTS using quadratic regression
def predict_pts_quadratic(model, poly, plays, yds):
    """Predicts PTS using quadratic regression."""
    # Create a DataFrame with the same feature names as used during fitting
    X_input = pd.DataFrame([[plays, yds]], columns=['total_snaps', 'yards_gained'])
    X_poly_input = poly.transform(X_input)
    pts = model.predict(X_poly_input)
    return pts[0]


# Function for user input and predictions
def main():
    # Parse command line arguments for the CSV file path
    parser = argparse.ArgumentParser(description="Predict PTS using Linear and Quadratic Regression")
    parser.add_argument('--file', type=str, default='dataframe.csv', help="Path to the CSV file")
    args = parser.parse_args()

    # Load the data
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    df = load_data(file_path)
    X, y = prepare_data(df)

    # Run the models
    model_linear, model_poly, r2_linear, r2_poly, poly = run_models(X, y)

    print(f'Linear Regression R^2 Score: {r2_linear:.4f}')
    print(f'Quadratic Regression R^2 Score: {r2_poly:.4f}')

    # Get user input
    plays = float(input("Enter number of plays: "))
    yds = float(input("Enter total yards: "))

    # Predict using linear regression
    pts_linear = predict_pts_linear(model_linear, plays, yds)
    print(f"Predicted PTS (Linear Regression): {pts_linear:.2f}")

    # Predict using quadratic regression
    pts_quadratic = predict_pts_quadratic(model_poly, poly, plays, yds)
    print(f"Predicted PTS (Quadratic Regression): {pts_quadratic:.2f}")


if __name__ == "__main__":
    main()
