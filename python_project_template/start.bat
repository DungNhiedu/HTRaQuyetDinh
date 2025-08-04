@echo off
echo 🚀 Stock Market Prediction System
echo ==================================

echo 📦 Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ❌ Virtual environment not found. Please run setup.sh first.
    pause
    exit /b 1
)

echo 🌐 Starting Streamlit application...
streamlit run src\stock_predictor\app.py

pause
