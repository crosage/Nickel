# Nickel

Paper Reader backend service for CVF paper sync, PDF extraction, LLM analysis, chunked translation, and figure reinsertion.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"
python server.py
```

The API listens on `0.0.0.0:8000` by default.

Runtime data is stored in `paper_reader_data/` next to `server.py` unless `PAPER_READER_DATA_DIR` is set.
