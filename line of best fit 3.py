import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D  # Importing for 3D plotting

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

# Quadratic regression
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)
r2_poly = r2_score(y, y_pred_poly)
print(f'Quadratic Regression R^2 Score: {r2_poly}')

# Plotting data for visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['total_snaps'], df['yards_gained'], y, color='black', label='Actual Data')

# Plot linear regression surface
X_surface, Y_surface = np.meshgrid(np.linspace(df['total_snaps'].min(), df['total_snaps'].max(), 100),
                                   np.linspace(df['yards_gained'].min(), df['yards_gained'].max(), 100))
Z_surface_linear = model_linear.intercept_ + model_linear.coef_[0] * X_surface + model_linear.coef_[1] * Y_surface
ax.plot_surface(X_surface, Y_surface, Z_surface_linear, color='blue', alpha=0.3, label='Linear Regression Surface')

# Plot quadratic regression surface
Z_surface_poly = model_poly.intercept_ + model_poly.coef_[0] * X_surface + model_poly.coef_[1] * Y_surface \
                 + model_poly.coef_[2] * X_surface**2 + model_poly.coef_[3] * X_surface * Y_surface + model_poly.coef_[4] * Y_surface**2
ax.plot_surface(X_surface, Y_surface, Z_surface_poly, color='red', alpha=0.3, label='Quadratic Regression Surface')

ax.set_xlabel('total_snaps')
ax.set_ylabel('yards_gained')
ax.set_zlabel('total_points')
ax.set_title('3D Plot of Linear and Quadratic Regression Surfaces')
plt.show()

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

# Select top features for inclusion
top_features = correlations.nlargest(7)
print("\nTop 7 Features based on correlation:")
print(top_features)
