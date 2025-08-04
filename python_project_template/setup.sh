#!/bin/bash

echo "🚀 Stock Market Prediction System - Quick Setup"
echo "=============================================="

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null
then
    echo "❌ Python is not installed. Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null
then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    PYTHON_CMD=python
    PIP_CMD=pip
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # macOS/Linux
    source venv/bin/activate
fi

# Install dependencies
echo "📚 Installing dependencies..."
$PIP_CMD install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 To start the application:"
echo "   streamlit run src/stock_predictor/app.py"
echo ""
echo "� Or run the main module:"
echo "   cd src && python -m stock_predictor.main"
echo ""
echo "�📖 For detailed instructions, see: GETTING_STARTED.md"
echo ""
