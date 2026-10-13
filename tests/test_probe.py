
"""Unit tests for the pure helpers (no media or ASR dependency)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_pipeline import pipeline as pl


def test_srt_timestamp_formats_milliseconds():
    assert pl.srt_timestamp(0) == "00:00:00,000"
    assert pl.srt_timestamp(3661.5) == "01:01:01,500"


def test_duration_of_missing_format_returns_zero():
    assert pl.duration_of({}) == 0.0
    assert pl.duration_of({"format": {"duration": "12.5"}}) == 12.5


def test_video_size_picks_first_video_stream():
    info = {"streams": [{"codec_type": "audio"}, {"codec_type": "video", "width": 1080, "height": 1920}]}
    assert pl.video_size(info) == (1080, 1920)


def test_iter_media_filters_by_extension(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "c.MOV").write_bytes(b"x")
    names = sorted(p.name for p in pl.iter_media(tmp_path))
    assert names == ["a.mp4", "c.MOV"]


def test_write_srt(tmp_path):
    segs = [{"start": 0.0, "end": 1.25, "text": " hello "}, {"start": 1.25, "end": 2.0, "text": "world"}]
    out = tmp_path / "x.srt"
    pl.write_srt(segs, out)
    body = out.read_text(encoding="utf-8")
    assert body.startswith("1\n00:00:00,000 --> 00:00:01,250\nhello")
    assert "\n2\n" in body
