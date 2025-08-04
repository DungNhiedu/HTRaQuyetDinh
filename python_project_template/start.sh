#!/bin/bash

echo "🚀 Stock Market Prediction System"
echo "=================================="

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Start the application
echo "🌐 Starting Streamlit application..."
streamlit run src/stock_predictor/app.py
