# Stochastic Gradient Descent (SGD) Regressor

A production-ready Python implementation of Linear Regression using Stochastic Gradient Descent. This project demonstrates modern Python best practices, including type hinting, logging, argument parsing, and automated build pipelines.

## 🏗 Architecture
The core logic is encapsulated in the `SGDRegressor` class, ensuring modularity and reusability.
- **Model**: Linear Regression ( = wx + b$)
- **Optimization**: Stochastic Gradient Descent (updates weights per sample)
- **Stack**: NumPy (Math), Matplotlib (Visualization)

## 🚀 Quick Start

### 1. Automated Build (Recommended)
The build script handles environment creation, dependency installation, linting, and execution.
```bash
./build.sh
```

### 2. Manual Execution
If you prefer running manually:
```bash
# Install
pip install -r requirements.txt

# Run with custom parameters
python main.py --epochs 100 --lr 0.05 --samples 500
```

## ⚙️ CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--epochs` | 50 | Number of training iterations over the dataset |
| `--lr` | 0.01 | Learning rate (step size) |
| `--samples` | 100 | Number of synthetic data points to generate |

## 🛡 Quality Control
This project enforces code quality using:
- **Black**: Deterministic code formatting.
- **MyPy**: Static type checking.
- **Pylint**: Code analysis and linting.
