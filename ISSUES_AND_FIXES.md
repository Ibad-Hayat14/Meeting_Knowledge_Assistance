# Issues Faced & Fixes Applied

This document details the issues encountered while trying to run the Meeting Knowledge Assistant and how they were resolved.

---

## 1. Global vs. Virtual Environment Python Paths
### **Issue Description**
When running `uvicorn src.api.main:app --reload` from the root directory, you encountered the following errors:
1. `uvicorn: command not found` (because it was not installed globally).
2. After installing `uvicorn` globally via `sudo apt install uvicorn`, running the app threw a `ModuleNotFoundError: No module named 'fastapi'`.

### **Root Cause**
The python dependencies (such as `fastapi`, `groq`, `streamlit`, and `chromadb`) were installed inside the virtual environment (`.venv`). When you ran `uvicorn` globally, it used the system-level Python interpreter which does not look inside `.venv/lib/` for packages.

### **Solution / Fix**
To run the app correctly without activation issues, you should execute the binaries directly from the `.venv/bin/` folder. This guarantees that Python loads the virtual environment and all its installed packages.

To make this extremely simple, we created two helper shell scripts:
1. **`run_backend.sh`**: Runs the FastAPI backend directly using `./.venv/bin/uvicorn`.
2. **`run_frontend.sh`**: Runs the Streamlit frontend directly using `./.venv/bin/streamlit`.

You can now start the applications by simply running:
```bash
# Terminal 1
./run_backend.sh

# Terminal 2
./run_frontend.sh
```

---

## 2. Port Address Conflicts (Address Already In Use)
### **Issue Description**
If another process is running on port `8000` (FastAPI default) or `8501` (Streamlit default), you will get an error like:
`[Errno 98] Address already in use`

### **Solution / Fix**
Ensure you stop any previous running instances of `uvicorn` or `streamlit` before starting the app. If you need to run on a different port, you can customize the port:
* For backend: `uvicorn src.api.main:app --reload --port <port_number>`
* For frontend (edit `API_BASE` in `src/ui/app.py` first): `streamlit run src/ui/app.py --server.port <port_number>`
