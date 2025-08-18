import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.beta0 = None  # Intercept
        self.beta1 = None  # Slope

    def fit(self, X, y):
        """
        Fit the linear regression model using least squares.
        Formula:
        β1 = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
        β0 = ȳ - β1 * x̄
        """
        X = np.array(X)
        y = np.array(y)

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Slope (β1)
        self.beta1 = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)

        # Intercept (β0)
        self.beta0 = y_mean - self.beta1 * x_mean

    def predict(self, X):
        """
        Predict target values using the trained model.
        Formula: y = β0 + β1 * x
        """
        X = np.array(X)
        return self.beta0 + self.beta1 * X

    def __str__(self):
        return f"y = {self.beta0:.2f} + {self.beta1:.2f}x"


if __name__ == "__main__":
    # Toy dataset
    X = [1, 2, 3, 4, 5]
    y = [52, 55, 59, 64, 68]

    # Train model
    model = SimpleLinearRegression()
    model.fit(X, y)

    print("Model:", model)
    print("Intercept (β0):", model.beta0)
    print("Slope (β1):", model.beta1)

    # Prediction
    X_new = [6, 7, 8]
    predictions = model.predict(X_new)
    print(f"Predictions for {X_new}:", predictions)
