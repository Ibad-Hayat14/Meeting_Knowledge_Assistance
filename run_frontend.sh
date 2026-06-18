#!/bin/bash
# Run Streamlit Frontend using the virtual environment streamlit
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment (.venv) not found in the current directory."
    echo "Please create it using: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "Starting Streamlit Frontend on http://localhost:8501..."
./.venv/bin/streamlit run src/ui/app.py
