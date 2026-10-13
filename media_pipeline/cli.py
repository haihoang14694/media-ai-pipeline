
"""CLI: python -m media_pipeline run --input DIR --out DIR [--aspects 9:16,1:1,16:9]"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import pipeline as pl


def process_asset(src: Path, out_root: Path, aspects: list[str], model: str, language, do_asr: bool, do_thumbs: bool, do_scenes: bool, height_cap):
    res = pl.AssetResult(name=src.stem, path=str(src), size_mb=src.stat().st_size / 1_048_576)
    work = out_root / src.stem
    work.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        info = pl.probe(src)
        res.probe_s = time.time() - t0
        res.duration_s = pl.duration_of(info)
        w, h = pl.video_size(info)
        (work / "probe.json").write_text(json.dumps({"width": w, "height": h, "duration_s": res.duration_s}, indent=2), encoding="utf-8")
        res.outputs.append(f"{src.stem}/probe.json")

        if do_asr:
            try:
                segments, elapsed, lang = pl.transcribe(src, model_size=model, language=language)
                res.asr_s, res.asr_segments = elapsed, len(segments)
                srt = work / f"{src.stem}.srt"
                pl.write_srt(segments, srt)
                txt = work / f"{src.stem}.txt"
                txt.write_text(" ".join(s["text"].strip() for s in segments), encoding="utf-8")
                (work / "asr_meta.json").write_text(json.dumps({"model": model, "language": lang, "segments": len(segments), "seconds": round(elapsed, 2)}, indent=2), encoding="utf-8")
                res.outputs += [f"{src.stem}/{srt.name}", f"{src.stem}/{txt.name}", f"{src.stem}/asr_meta.json"]
            except ImportError as e:
                res.errors.append(f"asr skipped: faster-whisper not installed ({e})")

        t0 = time.time()
        for aspect in aspects:
            dst = work / f"{src.stem}_{aspect.replace(':', 'x')}.mp4"
            pl.reframe(src, dst, aspect, height_cap=height_cap)
            res.outputs.append(f"{src.stem}/{dst.name}")
        res.renders_s = time.time() - t0

        if do_thumbs:
            t0 = time.time()
            sheet = work / f"{src.stem}_sheet.jpg"
            pl.thumbnails(src, sheet)
            res.thumbs_s = time.time() - t0
            res.outputs.append(f"{src.stem}/{sheet.name}")

        if do_scenes:
            t0 = time.time()
            cands = pl.scene_candidates(src)
            res.scenes_s = time.time() - t0
            res.scene_candidates = len(cands)
            (work / "scene_candidates.json").write_text(json.dumps({"threshold": 0.30, "count": len(cands), "times": [round(t, 2) for t in cands]}, indent=2), encoding="utf-8")
            res.outputs.append(f"{src.stem}/scene_candidates.json")
    except Exception as e:  # keep the run going, record the failure
        res.ok = False
        res.errors.append(f"{type(e).__name__}: {e}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="media-ai-pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process every video in a folder")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--aspects", default="9:16,1:1,16:9")
    r.add_argument("--model", default="tiny")
    r.add_argument("--language", default=None)
    r.add_argument("--no-asr", action="store_true")
    r.add_argument("--no-thumbs", action="store_true")
    r.add_argument("--no-scenes", action="store_true")
    r.add_argument("--height-cap", type=int, default=None)
    r.add_argument("--workers", type=int, default=2)

    p = sub.add_parser("probe", help="print ffprobe summary for one file")
    p.add_argument("file")

    a = ap.parse_args(argv)
    if a.cmd == "probe":
        info = pl.probe(Path(a.file))
        w, h = pl.video_size(info)
        print(json.dumps({"file": a.file, "width": w, "height": h, "duration_s": pl.duration_of(info)}, indent=2))
        return 0

    src_dir, out_root = Path(a.input), Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    files = pl.iter_media(src_dir)
    if not files:
        print(f"no video files in {src_dir}")
        return 2
    aspects = [x.strip() for x in a.aspects.split(",") if x.strip()]
    t_all = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = [
            pool.submit(process_asset, f, out_root, aspects, a.model, a.language,
                        not a.no_asr, not a.no_thumbs, not a.no_scenes, a.height_cap)
            for f in files
        ]
        for fut in futs:
            res = fut.result()
            results.append(res.as_dict())
            flag = "OK " if res.ok else "ERR"
            print(f"{flag} {res.name}: total {res.total_s}s (asr {res.asr_s}s, renders {res.renders_s}s, scenes {res.scenes_s}s)")
    manifest = {
        "tool": "media-ai-pipeline",
        "version": "0.1.0",
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "input_dir": str(src_dir),
        "aspects": aspects,
        "asr_model": None if a.no_asr else a.model,
        "assets": results,
        "totals": {
            "assets": len(results),
            "ok": sum(1 for r in results if r["ok"]),
            "wall_clock_s": round(time.time() - t_all, 2),
            "sum_asset_s": round(sum(r["total_s"] for r in results), 2),
            "asr_s": round(sum(r["timing_s"]["transcribe"] for r in results), 2),
            "renders_s": round(sum(r["timing_s"]["renders"] for r in results), 2),
            "scene_s": round(sum(r["timing_s"]["scene_scan"] for r in results), 2),
            "local_cost_usd": 0.0,
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    t = manifest["totals"]
    print(f"\nassets {t['assets']} (ok {t['ok']}) | wall {t['wall_clock_s']}s | asr {t['asr_s']}s | renders {t['renders_s']}s | scene {t['scene_s']}s | cost $0 (local)")
    print(f"manifest: {out_root / 'manifest.json'}")
    return 0 if t["ok"] == t["assets"] else 1


if __name__ == "__main__":
    sys.exit(main())
