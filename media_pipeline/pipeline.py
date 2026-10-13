
"""Core helpers for the media-ai-pipeline: probe, transcribe, reframe, thumbnails, scene candidates."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

TARGETS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1280, 720),
}


def _bin(name: str) -> str:
    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found on PATH and no ${name.upper()} override set")
    return found


def ffmpeg() -> str:
    return _bin("ffmpeg")


def ffprobe() -> str:
    return _bin("ffprobe")


def run(cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


@dataclass
class AssetResult:
    name: str
    path: str
    ok: bool = True
    duration_s: float = 0.0
    size_mb: float = 0.0
    probe_s: float = 0.0
    asr_s: float = 0.0
    asr_segments: int = 0
    renders_s: float = 0.0
    thumbs_s: float = 0.0
    scenes_s: float = 0.0
    scene_candidates: int = 0
    outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    local_cost_usd: float = 0.0

    @property
    def total_s(self) -> float:
        return self.probe_s + self.asr_s + self.renders_s + self.thumbs_s + self.scenes_s

    def as_dict(self) -> dict:
        return {
            "asset": self.name,
            "source": self.path,
            "ok": self.ok,
            "duration_s": round(self.duration_s, 2),
            "source_size_mb": round(self.size_mb, 2),
            "timing_s": {
                "probe": round(self.probe_s, 2),
                "transcribe": round(self.asr_s, 2),
                "renders": round(self.renders_s, 2),
                "thumbnails": round(self.thumbs_s, 2),
                "scene_scan": round(self.scenes_s, 2),
            },
            "total_s": round(self.total_s, 2),
            "asr_segments": self.asr_segments,
            "scene_candidates": self.scene_candidates,
            "outputs": self.outputs,
            "errors": self.errors,
            "local_cost_usd": self.local_cost_usd,
        }


def probe(path: Path) -> dict:
    code, out = run([
        ffprobe(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe failed for {path.name}: {out[-300:]}")
    return json.loads(out)


def duration_of(info: dict) -> float:
    try:
        return float(info.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def video_size(info: dict) -> tuple[int, int]:
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s.get("width") or 0), int(s.get("height") or 0)
    return 0, 0


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], out_path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def decode_audio(path: Path, sample_rate: int = 16000):
    """Decode any input to mono float32 PCM at `sample_rate` using ffmpeg.

    Done this way on purpose: faster-whisper's built-in decoder goes through PyAV, which
    breaks on some PyAV builds (`metadata_errors`). ffmpeg is already a hard dependency of
    this pipeline, so decoding through it removes a fragile link and keeps one audio path.
    """
    import numpy as np

    # binary-safe decode: text=True would corrupt the PCM stream
    proc = subprocess.run([
        ffmpeg(), "-v", "error", "-i", str(path),
        "-vn", "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-",
    ], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"audio decode failed: {proc.stderr[-200:]!r}")
    pcm = np.frombuffer(proc.stdout, dtype="<i2").astype("float32") / 32768.0
    return pcm


_ASR_CACHE: dict[str, object] = {}


def _load_model(model_size: str, device: str = "cpu", compute_type: str = "int8"):
    from faster_whisper import WhisperModel

    key = f"{model_size}:{device}:{compute_type}"
    if key not in _ASR_CACHE:
        _ASR_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _ASR_CACHE[key]


def transcribe(path: Path, model_size: str = "tiny", language: str | None = None) -> tuple[list[dict], float, str]:
    """Return (segments, elapsed_seconds, detected_language). Requires faster-whisper + numpy."""
    t0 = time.time()
    model = _load_model(model_size)
    audio = decode_audio(path)
    seg_iter, info = model.transcribe(audio, language=language, vad_filter=True)
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text}
        for s in seg_iter
    ]
    return segments, time.time() - t0, getattr(info, "language", language or "unknown")


def reframe(src: Path, dst: Path, aspect: str, height_cap: int | None = None) -> None:
    if aspect not in TARGETS:
        raise ValueError(f"unsupported aspect {aspect}")
    w, h = TARGETS[aspect]
    if height_cap:
        scale = height_cap / h
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    vf = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=24:3[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
    )
    code, out = run([
        ffmpeg(), "-y", "-i", str(src),
        "-filter_complex", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(dst),
    ])
    if code != 0 or not dst.exists():
        raise RuntimeError(f"reframe {aspect} failed: {out[-300:]}")


def thumbnails(src: Path, dst: Path, cols: int = 3, rows: int = 3) -> None:
    # 9 evenly spaced frames in a contact sheet + 1 poster frame
    code, out = run([
        ffmpeg(), "-y", "-i", str(src),
        "-vf", f"fps=1/ max(1\\,({cols*rows}))".replace(" ", ""),
        "-frames:v", "1", str(dst.with_suffix(".poster.jpg")),
    ])
    code, out = run([
        ffmpeg(), "-y", "-i", str(src),
        "-vf", f"select='not(mod(n\\,60))',scale=480:-1,tile={cols}x{rows}",
        "-frames:v", "1", str(dst),
    ])
    if code != 0 or not dst.exists():
        raise RuntimeError(f"thumbnails failed: {out[-300:]}")


def scene_candidates(src: Path, threshold: float = 0.30) -> list[float]:
    code, out = run([
        ffmpeg(), "-i", str(src),
        "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ])
    times: list[float] = []
    for line in out.splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                continue
    return times


def iter_media(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT)
