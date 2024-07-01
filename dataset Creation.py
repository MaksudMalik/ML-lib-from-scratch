import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate a multiple linear regression dataset
# Parameters
num_samples = 100
num_features = 3  # Number of features
true_intercept = 5.0
true_coefficients = np.array([2.0, -3.5, 1.5])  # Coefficients for each feature
noise_level = 1.0

# Generate features (X)
X = np.random.rand(num_samples, num_features) * 10  # Random values between 0 and 10

# Generate target variable (y) with some noise
y = true_intercept + X.dot(true_coefficients) + np.random.randn(num_samples) * noise_level

# Combine into a DataFrame
data = np.hstack((X, y.reshape(-1, 1)))
columns = [f'X{i+1}' for i in range(num_features)] + ['y']
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv('mlr_data.csv', index=False)

print("Dataset created and saved to 'mlr_data.csv'")
