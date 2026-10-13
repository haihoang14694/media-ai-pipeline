# media-ai-pipeline

Batch AI pipeline for short-form video: probe, transcribe, caption, reframe to vertical and square,
thumbnail sheet, scene-change candidates, and a JSON manifest with timing and cost per asset.

Built by Hoang Hai (video editor, Hanoi) because the same five steps were being repeated by hand for
every batch of clips. Everything here runs on ffmpeg plus Python, on CPU, with no paid API.

## What it does

One command processes a folder of clips:

```bash
# ffmpeg/ffprobe are picked up from PATH, or from $FFMPEG / $FFPROBE
python -m media_pipeline run --input ./clips --out ./out --aspects 9:16,1:1 --model base --workers 2 --height-cap 720
```

Per asset it writes:

- `probe.json` - width, height, duration
- `<name>.srt` and `<name>.txt` - transcription (faster-whisper, VAD filtered)
- `asr_meta.json` - model, detected language, segment count, transcription seconds
- `<name>_9x16.mp4`, `<name>_1x1.mp4` - reframes with blurred-background padding, H.264, faststart
- `<name>_sheet.jpg` + `<name>_sheet.poster.jpg` - contact sheet and poster frame
- `scene_candidates.json` - scene-change timestamps (ffmpeg `scene` filter, threshold 0.30)
- a row in `out/manifest.json` with per-stage timings, output list, errors and local cost

```
python -m media_pipeline probe file.mp4     # single file metadata
python -m pytest -q tests/                  # unit tests for the pure helpers
bash demo_run.sh /path/to/clips             # real run on sample clips
```

## Measured run (real numbers, not estimates)

4 clips, 331.5 s of source video (two 157.7 s reels + two 8 s generated clips), ASR model `base`,
CPU int8, 2 workers, renders capped at 720p, aspects 9:16 + 1:1.

| Metric | Value |
|---|---|
| Assets processed / failed | 4 / 0 |
| Wall clock | 107.97 s |
| Sum of per-asset time | 204.94 s |
| Transcription (total) | 58.58 s |
| Renders (total, 8 files) | 132.78 s |
| Scene scan (total) | 10.07 s |
| Local API cost | $0.00 |
| Throughput | ~20 s of compute per minute of source video |
| Per clip | 6.85 s to 97.0 s depending on length and speech density |
| Scene candidates | 32 per 157.7 s reel at threshold 0.30 |

Environment: Python 3.14.7, Linux x86_64, ffmpeg 9.0.1, faster-whisper `base`, CPU int8, 2 workers.
Raw output of that run is in `demo_output_full/manifest.json`.

## Design decisions

- **Audio path avoids PyAV.** faster-whisper's built-in decoder goes through PyAV, which fails on some
  PyAV builds (`TypeError: open() got an unexpected keyword argument 'metadata_errors'`). The pipeline
  decodes audio itself: `ffmpeg -f s16le -ac 1 -ar 16000` into a float32 array. ffmpeg is already a hard
  dependency, so this removes a fragile link instead of pinning around it. The failure and the fix are
  both in the git history.
- **A failing asset does not kill the batch.** Each asset is processed independently, errors are recorded
  in its manifest row, and the exit code reflects only whether all assets succeeded.
- **Manifest first.** Timing and cost are recorded per stage so a slow stage is visible instead of guessed.
- **CPU, int8, no paid API.** The pipeline has to be runnable on a laptop with no budget.

## Limitations (what is not finished)

- **Captions need speech.** The two reels in the demo run are music-led, so VAD returned 0 segments and the
  caption files are empty. That is correct behaviour, not a failed step: with dialogue (the lip-sync sample)
  the same path produces segments and a filled SRT.
- **No burn-in captions, no translation, no speaker labels** yet. Output is sidecar SRT only.
- **Scene candidates are timestamps only.** There is no ranking or automatic B-roll selection yet.
- **CPU only.** No CUDA path and no batching across assets; throughput is one asset at a time per worker.
- **Cost comparison is local-only.** Hosted ASR or generative video APIs are not wired in, so the
  quality/speed/cost comparison against paid models cannot be produced from this repo alone.
- **Not in this repo yet:** n8n orchestration, Shopify/Meta integrations, monitoring dashboard, RAG.

## Roadmap

1. n8n workflow: webhook -> this pipeline -> Google Sheets log -> Telegram notification.
2. Ranking pass over scene candidates (shot quality, motion, face presence).
3. Dashboard over `manifest.json` (assets, minutes saved, cost per asset, error rate).
4. Optional hosted-ASR backend to measure the quality/speed/cost tradeoff against the local model.

## Requirements

- Python 3.10+, `ffmpeg` and `ffprobe` on PATH (or `$FFMPEG` / `$FFPROBE`)
- `pip install -r requirements.txt` (faster-whisper; numpy comes with it)
- Tests need `pytest`

## License

MIT - see `LICENSE`.
