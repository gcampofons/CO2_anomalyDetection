#!/bin/bash
# Automatic setup script for Linux/Mac
# CO2 Anomaly Detection Project

echo "========================================"
echo " CO2 Anomaly Detection - Setup"
echo "========================================"
echo ""

# Verify Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.13"
    exit 1
fi

echo "[1/5] Verifying Python version..."
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Check if Python 3.13 is installed
if [[ ! $PYTHON_VERSION =~ ^3\.13 ]]; then
    echo "[ERROR] Python 3.13 is required, but found $PYTHON_VERSION"
    echo "Please install Python 3.13 from https://www.python.org/downloads/"
    exit 1
fi

echo "[OK] Python 3.13 detected"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created successfully."
else
    echo "[2/5] Virtual environment already exists."
fi
echo ""

# Activate virtual environment and upgrade pip
echo "[3/5] Activating virtual environment and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip --quiet
echo ""

# Install dependencies
echo "[4/5] Installing dependencies..."
pip install -r requirements.txt
echo ""

# Verify installation
echo "[5/5] Verifying installation..."
python3 -c "import tensorflow, numpy, pandas, sklearn; print('All packages installed successfully!')"
echo ""
echo "========================================"
echo " Installation completed!"
echo "========================================"
echo ""
echo "To run the project:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run the script: python AE_LSTM_anomaly_detection.py"
echo ""
echo "IMPORTANT: Download the dataset CEIP_Albea_ValldAlba.csv"
echo "from https://doi.org/10.5281/zenodo.5036228"
echo ""
