import numpy as np
from linear_regression import LinearRegressionScratch

# Testing with a new dataset
X_test = np.array([10, 20, 30, 40, 50])
y_test = np.array([15, 25, 35, 45, 55])  # perfectly linear: y = x + 5

model = LinearRegressionScratch().fit(X_test, y_test)

print("Test Dataset Model Parameters:", model.get_params())

# Predictions
for x in [60, 70, 80]:
    print(f"Prediction for x={x}: {model.predict(x):.2f}")