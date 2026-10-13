#!/usr/bin/env bash
# Real demo: process sample clips end to end and write demo_output/manifest.json
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export FFMPEG="${FFMPEG:-/home/local-sever/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg}"
export FFPROBE="${FFPROBE:-/home/local-sever/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffprobe}"
INPUT="${1:?usage: demo_run.sh INPUT_DIR}"
OUT="${2:-$HERE/demo_output}"
export PATH="/home/local-sever/.hermes/tools/uv-0.12.3-linux-x64:$PATH"
unset PYTHONPATH
cd "$HERE"
uv run --no-project --with faster-whisper -- python -m media_pipeline run --input "$INPUT" --out "$OUT" --aspects 9:16,1:1 --model tiny --workers 2 --height-cap 720
