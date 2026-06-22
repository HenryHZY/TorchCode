# Start Standalone Web UI

## First-Time Setup

Run these from the project root:

```bash
python -m pip install -r api/requirements.txt
cd web && npm install
```

## Start Backend

Terminal 1, from the project root:

```bash
python -m uvicorn api.main:app --port 8000 --reload
```

## Start Frontend

Terminal 2, from the project root:

```bash
cd web && WATCHPACK_POLLING=true NEXT_TELEMETRY_DISABLED=1 npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000/
```

## Stop Servers

Press `Ctrl+C` in both terminals, or run:

```bash
lsof -tiTCP:3000 -sTCP:LISTEN | xargs -r kill
lsof -tiTCP:8000 -sTCP:LISTEN | xargs -r kill
```

## Quick Checks

```bash
curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:8000/docs
```
