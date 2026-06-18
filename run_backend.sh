#!/bin/bash
# Run FastAPI Backend using the virtual environment uvicorn
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment (.venv) not found in the current directory."
    echo "Please create it using: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "Starting FastAPI Backend on http://localhost:8000..."
./.venv/bin/uvicorn src.api.main:app --reload
