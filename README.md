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

## Deploy

This server is currently deployed as `paper-reader.service` from this repo:

```bash
sudo install -d -m 750 -o server118 -g server118 /etc/nickel
sudo install -m 600 -o server118 -g server118 deploy/env/nickel.env.example /etc/nickel/nickel.env
sudo install -m 644 deploy/systemd/paper-reader.service /etc/systemd/system/paper-reader.service
sudo systemctl daemon-reload
sudo systemctl enable --now paper-reader.service
```

Edit `/etc/nickel/nickel.env` with the real LLM key before starting the service.

Cloudflare Tunnel exposes the backend through the existing `pixiv-helper` tunnel:

```bash
cloudflared tunnel route dns pixiv-helper paper-api.zundamon.bond
sudo install -m 600 deploy/cloudflared/config.yml.example /etc/cloudflared/config.yml
sudo cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate
sudo systemctl restart cloudflared.service
```

Production URL:

```text
https://paper-api.zundamon.bond
```
