#!/bin/bash

# Legal Document Chat - Local Run Script
# Quick script to run the Streamlit application locally

echo "🚀 Starting Legal Document Chat Application..."
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file"
        echo "⚠️  Please edit .env and add your API keys before running again"
        exit 1
    else
        echo "❌ .env.example not found!"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Run Streamlit app
echo "🌐 Launching Streamlit app..."
echo "📍 App will open at: http://localhost:8501"
echo ""
streamlit run app.py
