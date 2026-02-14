#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting Production Build Pipeline..."

# --- Step 1: Environment Setup ---
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# --- Step 2: Install Dependencies ---
echo "⬇️  Installing Dependencies..."
pip install -r requirements.txt --quiet
echo "   Dependencies installed."

# --- Step 3: Code Quality Checks ---
echo "🎨 Running Black (Formatter)..."
black main.py

echo "🔍 Running MyPy (Type Checker)..."
mypy main.py --ignore-missing-imports

echo "🧐 Running Pylint (Linter)..."
# We disable specific warnings to keep the output clean for this demo
pylint main.py --disable=C0103,C0114,R0903 --score=n

# --- Step 4: Execution ---
echo "✅ Build & Checks Passed!"
echo "📈 Running Application with default args..."
python main.py --epochs 50 --samples 150
