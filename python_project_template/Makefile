# Stock Market Prediction System - Makefile

.PHONY: help install run clean test lint format setup dev

# Default target
help:
	@echo "Available commands:"
	@echo "  setup     - Setup the development environment"
	@echo "  install   - Install dependencies"
	@echo "  run       - Run the Streamlit application"
	@echo "  dev       - Run in development mode with auto-reload"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linting"
	@echo "  format    - Format code"
	@echo "  clean     - Clean up temporary files"

# Setup development environment
setup:
	@echo "Setting up development environment..."
	python -m pip install --upgrade pip
	pip install -e .
	pip install -r requirements-dev.txt

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Run the Streamlit application
run:
	@echo "Starting Stock Market Prediction System..."
	streamlit run src/stock_predictor/app.py --server.port 8501

# Run in development mode
dev:
	@echo "Starting development server with auto-reload..."
	streamlit run src/stock_predictor/app.py --server.port 8501 --server.fileWatcherType auto

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Run linting
lint:
	@echo "Running linting..."
	flake8 src/
	mypy src/

# Format code
format:
	@echo "Formatting code..."
	black src/
	black tests/

# Clean up temporary files
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -f streamlit.log

# Quick start (install and run)
start: install run

# Development setup and run
dev-setup: setup dev