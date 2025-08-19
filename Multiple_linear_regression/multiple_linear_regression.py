import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.beta = None  # coefficients (including intercept)

    def fit(self, X, y):
        """
        Fit multiple linear regression using the normal equation.
        Formula:
        β = (X^T X)^(-1) X^T y
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # add intercept column (1s)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])  # shape (n, p+1)

        # normal equation
        self.beta = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)

    def predict(self, X):
        """
        Predict target values using trained model.
        Formula: y = β0 + β1*x1 + β2*x2 + ...
        """
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)  # single row
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])
        return X_b @ self.beta

    def __str__(self):
        terms = [f"{self.beta[0]:.2f}"]
        for i, b in enumerate(self.beta[1:], start=1):
            terms.append(f"{b:.2f}*x{i}")
        return "y = " + " + ".join(terms)


if __name__ == "__main__":
    # Example dataset: Rent predicted by [Area, Bedrooms]
    X = [
        [7, 1],
        [8, 2],
        [10, 2],
        [12, 3]
    ]
    y = [1200, 1500, 1700, 2100]

    # Train model
    model = MultipleLinearRegression()
    model.fit(X, y)

    print("Model:", model)
    print("Coefficients (β):", model.beta)

    # Predictions
    X_new = [[9, 2], [11, 3]]
    predictions = model.predict(X_new)
    print(f"Predictions for {X_new}:", predictions)
