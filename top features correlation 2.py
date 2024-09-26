import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load CSV data into DataFrame using the full file path
df = pd.read_csv(r"C:\Users\pcpin\Desktop\sport gambling\dataframe.csv")

# Define features and target variable
X = df[['total_snaps', 'yards_gained']]  # Using 'total_snaps' and 'yards_gained' as features
y = df['total_points']  # Target variable

# Compute correlations
correlations = X.corrwith(y).abs()
print("Feature correlations with target:")
print(correlations)

# Plot correlations
plt.figure(figsize=(8, 5))
correlations.plot(kind='bar', color='skyblue')
plt.title('Feature Correlations with Target')
plt.xlabel('Feature')
plt.ylabel('Correlation')
plt.show()

# Prepare polynomial features (quadratic)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Linear regression
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)
r2_linear = r2_score(y, y_pred_linear)
print(f'Linear Regression R^2 Score: {r2_linear}')

# Quadratic regression
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)
r2_poly = r2_score(y, y_pred_poly)
print(f'Quadratic Regression R^2 Score: {r2_poly}')

# Select top features for inclusion
top_features = correlations.nlargest(7)
print("\nTop 7 Features based on correlation:")
print(top_features)
