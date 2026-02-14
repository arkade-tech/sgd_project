#!/bin/bash

echo "🚀 Starting Build Process..."

# 1. Create Virtual Environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 2. Install Dependencies
echo "⬇️ Installing dependencies..."
./venv/bin/pip install -r requirements.txt

echo "🎉 Build Complete! Run the app with: ./venv/bin/python main.py"
