import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load CSV data into DataFrame using the full file path
df = pd.read_csv(r"C:\Users\pcpin\Desktop\sport gambling\dataframe.csv")

# Define features and target variable
X = df[['total_snaps', 'yards_gained']]  # Adjusting feature names to match the CSV file
y = df['total_points']  # Target variable, adjust based on your CSV file's column names

# Prepare polynomial features (quadratic)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Linear regression
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)
r2_linear = r2_score(y, y_pred_linear)
print(f'Linear Regression R^2 Score: {r2_linear}')

# Get coefficients from trained linear regression model
linear_intercept = model_linear.intercept_
linear_coef = model_linear.coef_

# Quadratic regression
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)
r2_poly = r2_score(y, y_pred_poly)
print(f'Quadratic Regression R^2 Score: {r2_poly}')

# Get coefficients from trained quadratic regression model
quad_intercept = model_poly.intercept_
quad_coef = model_poly.coef_

# Function to predict PTS using linear regression
def predict_pts_linear(plays, yds):
    """Predicts PTS using linear regression."""
    pts = linear_intercept + linear_coef[0] * plays + linear_coef[1] * yds
    return pts

# Function to predict PTS using quadratic regression
def predict_pts_quadratic(plays, yds):
    """Predicts PTS using quadratic regression."""
    pts = (quad_intercept +
           quad_coef[0] * plays +
           quad_coef[1] * yds +
           quad_coef[2] * plays**2 +
           quad_coef[3] * plays * yds +
           quad_coef[4] * yds**2)
    return pts

# Function for user input and predictions
def main():
    # Get user input
    plays = float(input("Enter number of plays: "))
    yds = float(input("Enter total yards: "))

    # Predict using linear regression
    pts_linear = predict_pts_linear(plays, yds)
    print(f"Predicted PTS (Linear Regression): {pts_linear:.2f}")

    # Predict using quadratic regression
    pts_quadratic = predict_pts_quadratic(plays, yds)
    print(f"Predicted PTS (Quadratic Regression): {pts_quadratic:.2f}")

if __name__ == "__main__":
    main()
