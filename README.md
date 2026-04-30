# qwen3-tts-docker

Local Dockerized voice cloning app using Qwen3-TTS + Gradio (GPU/NVIDIA required).

## Requirements
- NVIDIA GPU + working NVIDIA Container Toolkit
- Docker + Docker Compose

## Quick Start
1. Clone this repo.
2. (Optional) create local TLS files in `certs/`.
3. Run:

```bash
docker compose up --build
```

App starts on `http://localhost:7860` (or HTTPS if cert files are present and mapped).

## Files
- `app.py`: Gradio app and Qwen3-TTS inference
- `Dockerfile`: CUDA + Python runtime image
- `docker-compose.yml`: service config and GPU runtime
