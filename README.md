# Stochastic Gradient Descent (SGD) Regressor

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Type Checker](https://img.shields.io/badge/types-mypy-blue)

A production-grade Python implementation of Linear Regression using Stochastic Gradient Descent (SGD). This project demonstrates modern software engineering practices, including static type checking, automated CI/CD pipelines, structured logging, and modular object-oriented design.

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Architecture](#-project-architecture)
- [Development Workflow](#-development-workflow)

---

## 🧐 Overview
Standard Gradient Descent updates weights using the entire dataset, which is computationally expensive. **Stochastic Gradient Descent (SGD)** updates weights using a single random sample at a time, making it much faster for large datasets, though deeper into the "noise."

This project implements SGD from scratch (using only NumPy) to fit a linear model:
161627 y = wx + b 161627

---

## 🌟 Key Features
- **Strict Type Hinting**: Full `mypy` compliance for type safety.
- **Automated Build Pipeline**: `build.sh` handles venv creation, dependency installation, linting, and testing.
- **Production Logging**: Uses Python's `logging` module instead of `print` statements.
- **CLI Interface**: Robust argument parsing using `argparse`.
- **Code Quality**: Enforced via `black` (formatting) and `pylint` (linting).

---

## 🚀 Installation & Setup

### Option A: The "One-Click" Build (Recommended)
We provide a build script that automates the entire setup process, acting as a local CI/CD pipeline.

```bash
# Makes the script executable (only needed once)
chmod +x build.sh

# Runs setup, tests, and execution
./build.sh
```

### Option B: Manual Setup
```bash
# 1. Create and activate environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python main.py
```

---

## 💻 Usage

You can customize the training process using Command Line Arguments:

```bash
python main.py --epochs 100 --lr 0.005 --samples 500
```

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--epochs` | `50` | Number of complete passes through the dataset. |
| `--lr` | `0.01` | Learning Rate (Step size for weight updates). |
| `--samples` | `100` | Number of synthetic data points to generate. |

---

## 🏗 Project Architecture

```text
sgd_project/
├── main.py            # Core logic (SGDRegressor class & CLI entry point)
├── requirements.txt   # Production & Dev dependencies
├── build.sh           # Automated build & quality check script
├── README.md          # Documentation
└── .gitignore         # Git configuration
```

### The `SGDRegressor` Class
The logic is encapsulated in a reusable class structure:
- **`fit(X, y)`**: Trains the model using the SGD algorithm.
- **`predict(X)`**: Generates predictions based on learned weights.
- **State**: Maintains `weights`, `bias`, and `loss_history`.

---

## 🛠 Development Workflow
This project enforces high code quality standards. Before pushing code, run the build script to ensure all checks pass:

1. **Formatting**: `black main.py`
2. **Type Checking**: `mypy main.py`
3. **Linting**: `pylint main.py`

