#!/usr/bin/env python3
"""
Stochastic Gradient Descent (SGD) Demonstration.

This script implements a custom Linear Regression model using Stochastic
Gradient Descent. It generates synthetic data, trains the model, and
visualizes the results.

Usage:
    python main.py --epochs 100 --lr 0.01 --samples 200
"""

import argparse
import logging
import sys
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SGDRegressor:
    """
    Linear Regression model trained using Stochastic Gradient Descent.

    Attributes:
        learning_rate (float): The step size for weight updates.
        n_epochs (int): Number of passes through the training dataset.
        weights (np.ndarray): Coefficents of the model (Slope/m).
        bias (np.ndarray): Intercept of the model (b).
        loss_history (List[float]): History of MSE loss per epoch.
    """

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 50):
        """
        Initializes the SGD Regressor.

        Args:
            learning_rate: Step size for the optimizer.
            n_epochs: Number of training iterations.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # Initialize as None, but type hint allows np.ndarray
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model using Stochastic Gradient Descent.

        Args:
            X: Training data features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 1).
        """
        n_samples, n_features = X.shape

        # Initialize parameters randomly as (1, 1) arrays for matrix math
        np.random.seed(42)
        self.weights = np.random.randn(n_features, 1)
        self.bias = np.random.randn(1, 1)  # Fixed: Initialize as 2D array

        logger.info(f"Starting training for {self.n_epochs} epochs...")

        for epoch in range(1, self.n_epochs + 1):
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            for i in range(n_samples):
                xi = X_shuffled[i : i + 1]  # Shape (1, n_features)
                yi = y_shuffled[i : i + 1]  # Shape (1, 1)

                # Safe check for Mypy (though fit initializes them)
                if self.weights is None or self.bias is None:
                    continue

                # Forward pass
                prediction = np.dot(xi, self.weights) + self.bias
                error = prediction - yi

                # Accumulate loss (MSE)
                epoch_loss += (error**2).item()

                # Backward pass (Gradients)
                grad_w = 2 * xi.T.dot(error)
                grad_b = 2 * error  # Shape (1,1)

                # Update weights
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # Log progress
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}/{self.n_epochs} | Loss: {avg_loss:.4f}")

        logger.info("Training completed.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for given input samples.

        Args:
            X: Input features.

        Returns:
            Predicted values.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        return np.dot(X, self.weights) + self.bias


def generate_synthetic_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic linear data with Gaussian noise."""
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    # True relation: y = 3x + 4 + noise
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    return X, y


def plot_results(X: np.ndarray, y: np.ndarray, model: SGDRegressor) -> None:
    """Visualizes the regression line and loss history."""
    try:
        plt.figure(figsize=(12, 5))

        # Plot 1: Regression
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, color="blue", alpha=0.5, label="Data")
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        plt.plot(
            x_range, model.predict(x_range), color="red", linewidth=2, label="SGD Fit"
        )
        plt.title("Linear Regression Fit")
        plt.xlabel("Feature X")
        plt.ylabel("Target y")
        plt.legend()

        # Plot 2: Loss History
        plt.subplot(1, 2, 2)
        plt.plot(model.loss_history, color="green")
        plt.title("Training Loss (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot results: {e}")


def main() -> None:
    """Main execution entry point."""
    parser = argparse.ArgumentParser(description="Train SGD Linear Regression.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of data samples"
    )

    args = parser.parse_args()

    # 1. Prepare Data
    logger.info(f"Generating {args.samples} synthetic samples...")
    X, y = generate_synthetic_data(args.samples)

    # 2. Initialize and Train Model
    model = SGDRegressor(learning_rate=args.lr, n_epochs=args.epochs)
    model.fit(X, y)

    # 3. Output Results
    # Assert ensures MyPy knows these aren't None
    assert model.weights is not None
    assert model.bias is not None

    logger.info(f"Final Model: y = {model.weights[0][0]:.2f}x + {model.bias[0][0]:.2f}")

    # 4. Visualize
    plot_results(X, y, model)


if __name__ == "__main__":
    main()
