#!/usr/bin/env python3
"""
Auto-detect a username template on screen and toggle an OBS blur source.

This file supports two modes:
1) CLI mode for local tuning/debug.
2) OBS script mode (when loaded by OBS, which imports this file with `obspython`).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from mss import mss
except ImportError:  # pragma: no cover - runtime dependency guard
    mss = None

try:
    import obspython as obs  # type: ignore
except Exception:  # pragma: no cover - only available inside OBS
    obs = None


DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "assets" / "template.png"
DEFAULT_SCALES = (0.95, 1.0, 1.05)


@dataclass
class MatchResult:
    found: bool
    score: float
    box: Optional[Tuple[int, int, int, int]] = None


def _parse_scales(raw: str | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(raw, str):
        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            values.append(float(token))
    else:
        values = [float(v) for v in raw]

    clean = []
    for v in values:
        if v <= 0:
            continue
        clean.append(v)

    if not clean:
        return DEFAULT_SCALES
    return tuple(sorted(set(clean)))


class FastTemplateMatcher:
    def __init__(
        self,
        template_path: str,
        threshold: float = 0.83,
        scales: Sequence[float] = DEFAULT_SCALES,
        downscale: float = 1.0,
    ) -> None:
        self.template_path = template_path
        self.threshold = float(threshold)
        self.scales = _parse_scales(scales)
        self.downscale = max(0.25, min(float(downscale), 1.0))
        self._variants: list[tuple[float, np.ndarray]] = []
        self._load_templates()

    def _load_templates(self) -> None:
        template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Template not found or unreadable: {self.template_path}")

        if self.downscale != 1.0:
            template = cv2.resize(
                template,
                dsize=None,
                fx=self.downscale,
                fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )

        variants: list[tuple[float, np.ndarray]] = []
        for scale in self.scales:
            if scale == 1.0:
                scaled = template
            else:
                scaled = cv2.resize(
                    template,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )

            h, w = scaled.shape[:2]
            if h < 4 or w < 4:
                continue
            variants.append((scale, scaled))

        if not variants:
            raise RuntimeError("No usable template variants were produced.")

        self._variants = variants

    def detect(self, frame_bgr: np.ndarray) -> MatchResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale != 1.0:
            gray = cv2.resize(
                gray,
                dsize=None,
                fx=self.downscale,
                fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )

        best_score = -1.0
        best_box: Optional[Tuple[int, int, int, int]] = None

        for _, templ in self._variants:
            t_h, t_w = templ.shape[:2]
            g_h, g_w = gray.shape[:2]
            if t_h > g_h or t_w > g_w:
                continue

            result = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            score = float(max_val)

            if score > best_score:
                x, y = max_loc
                best_score = score
                if self.downscale != 1.0:
                    x = int(round(x / self.downscale))
                    y = int(round(y / self.downscale))
                    t_w = int(round(t_w / self.downscale))
                    t_h = int(round(t_h / self.downscale))
                best_box = (x, y, t_w, t_h)

        return MatchResult(found=best_score >= self.threshold, score=max(best_score, 0.0), box=best_box)


class ScreenGrabber:
    def __init__(
        self,
        monitor: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if mss is None:
            raise RuntimeError("`mss` is required. Install dependencies from requirements.txt.")

        self._ctx = mss()
        self.monitor_index = monitor
        self.region = region
        self._target = self._resolve_target()

    def _resolve_target(self) -> dict:
        if self.monitor_index >= len(self._ctx.monitors):
            raise ValueError(
                f"Monitor {self.monitor_index} is out of range. Available: 1..{len(self._ctx.monitors)-1}"
            )
        base = self._ctx.monitors[self.monitor_index]
        if self.region is None:
            return base

        x, y, w, h = self.region
        if w <= 0 or h <= 0:
            raise ValueError("Region width/height must be > 0.")
        return {
            "left": base["left"] + int(x),
            "top": base["top"] + int(y),
            "width": int(w),
            "height": int(h),
        }

    def grab_bgr(self) -> np.ndarray:
        shot = self._ctx.grab(self._target)
        frame = np.array(shot, dtype=np.uint8)
        return frame[:, :, :3]


def _clamp_region(region: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = region
    if w <= 0 or h <= 0:
        return None
    return max(0, x), max(0, y), w, h


def _draw_box(frame_bgr: np.ndarray, box: Optional[Tuple[int, int, int, int]], score: float) -> np.ndarray:
    out = frame_bgr.copy()
    if box is None:
        return out
    x, y, w, h = box
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 230, 80), 2)
    cv2.putText(
        out,
        f"{score:.3f}",
        (x, max(16, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 80),
        2,
        cv2.LINE_AA,
    )
    return out


def run_cli(args: argparse.Namespace) -> int:
    matcher = FastTemplateMatcher(
        template_path=args.template,
        threshold=args.threshold,
        scales=_parse_scales(args.scales),
        downscale=args.downscale,
    )
    grabber = ScreenGrabber(monitor=args.monitor, region=_clamp_region((args.x, args.y, args.w, args.h)))

    print("Running detection. Press Ctrl+C to stop.")
    last_log = 0.0
    frame_interval = 1.0 / max(args.fps, 1)

    while True:
        start = time.perf_counter()
        frame = grabber.grab_bgr()
        result = matcher.detect(frame)

        now = time.time()
        if result.found and (now - last_log) > 0.25:
            print(f"[MATCH] score={result.score:.3f} box={result.box}")
            last_log = now

        if args.debug:
            view = _draw_box(frame, result.box, result.score)
            cv2.imshow("auto_blur_debug", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        elapsed = time.perf_counter() - start
        delay = frame_interval - elapsed
        if delay > 0:
            time.sleep(delay)

    cv2.destroyAllWindows()
    return 0


# ---------------------------
# OBS script integration
# ---------------------------

_obs_template_path = str(DEFAULT_TEMPLATE_PATH)
_obs_blur_source = ""
_obs_threshold = 0.83
_obs_monitor = 1
_obs_region = (0, 0, 0, 0)
_obs_interval_ms = 120
_obs_downscale = 1.0
_obs_scales = "0.95,1.0,1.05"
_obs_hits_required = 2
_obs_misses_before_hide = 4
_obs_debug_logs = False

_obs_matcher: Optional[FastTemplateMatcher] = None
_obs_grabber: Optional[ScreenGrabber] = None
_obs_visible = False
_obs_hits = 0
_obs_misses = 0


def _obs_log(level: int, msg: str) -> None:
    if obs is not None:
        obs.script_log(level, msg)


def _obs_set_source_visible(source_name: str, visible: bool) -> None:
    if obs is None or not source_name:
        return

    scene_source = obs.obs_frontend_get_current_scene()
    if scene_source is None:
        return
    scene = obs.obs_scene_from_source(scene_source)
    if scene is None:
        obs.obs_source_release(scene_source)
        return

    scene_item = None
    if hasattr(obs, "obs_scene_find_source_recursive"):
        scene_item = obs.obs_scene_find_source_recursive(scene, source_name)
    elif hasattr(obs, "obs_scene_find_source"):
        scene_item = obs.obs_scene_find_source(scene, source_name)

    if scene_item is not None:
        obs.obs_sceneitem_set_visible(scene_item, visible)

    obs.obs_source_release(scene_source)


def _obs_rebuild_runtime() -> None:
    global _obs_matcher, _obs_grabber
    global _obs_hits, _obs_misses, _obs_visible

    _obs_hits = 0
    _obs_misses = 0
    _obs_visible = False

    try:
        _obs_matcher = FastTemplateMatcher(
            template_path=_obs_template_path,
            threshold=_obs_threshold,
            scales=_parse_scales(_obs_scales),
            downscale=_obs_downscale,
        )
        _obs_grabber = ScreenGrabber(
            monitor=_obs_monitor,
            region=_clamp_region(_obs_region),
        )
        _obs_log(obs.LOG_INFO, "Auto-blur matcher ready.")
    except Exception as exc:  # pragma: no cover - OBS runtime only
        _obs_matcher = None
        _obs_grabber = None
        if obs is not None:
            _obs_log(obs.LOG_WARNING, f"Auto-blur disabled: {exc}")


def _obs_tick() -> None:
    global _obs_hits, _obs_misses, _obs_visible
    if _obs_matcher is None or _obs_grabber is None:
        return

    try:
        frame = _obs_grabber.grab_bgr()
        result = _obs_matcher.detect(frame)
    except Exception as exc:  # pragma: no cover - OBS runtime only
        _obs_log(obs.LOG_WARNING, f"Capture failure: {exc}")
        return

    if result.found:
        _obs_hits += 1
        _obs_misses = 0
    else:
        _obs_hits = 0
        _obs_misses += 1

    target_visible = _obs_visible
    if _obs_hits >= _obs_hits_required:
        target_visible = True
    elif _obs_visible and _obs_misses >= _obs_misses_before_hide:
        target_visible = False

    if target_visible != _obs_visible:
        _obs_visible = target_visible
        _obs_set_source_visible(_obs_blur_source, _obs_visible)
        if _obs_debug_logs:
            state = "ON" if _obs_visible else "OFF"
            _obs_log(obs.LOG_INFO, f"Blur {state} (score={result.score:.3f})")


def script_description() -> str:
    return (
        "Detect a username template on screen and toggle a blur source.\n"
        "Place your cropped username image at assets/template.png.\n"
        "Requires: opencv-python, numpy, mss."
    )


def script_defaults(settings) -> None:
    if obs is None:
        return
    obs.obs_data_set_default_string(settings, "template_path", str(DEFAULT_TEMPLATE_PATH))
    obs.obs_data_set_default_string(settings, "blur_source", "")
    obs.obs_data_set_default_double(settings, "threshold", 0.83)
    obs.obs_data_set_default_int(settings, "monitor", 1)
    obs.obs_data_set_default_int(settings, "x", 0)
    obs.obs_data_set_default_int(settings, "y", 0)
    obs.obs_data_set_default_int(settings, "w", 0)
    obs.obs_data_set_default_int(settings, "h", 0)
    obs.obs_data_set_default_int(settings, "interval_ms", 120)
    obs.obs_data_set_default_double(settings, "downscale", 1.0)
    obs.obs_data_set_default_string(settings, "scales", "0.95,1.0,1.05")
    obs.obs_data_set_default_int(settings, "hits_required", 2)
    obs.obs_data_set_default_int(settings, "misses_before_hide", 4)
    obs.obs_data_set_default_bool(settings, "debug_logs", False)


def script_properties():
    if obs is None:
        return None
    props = obs.obs_properties_create()
    obs.obs_properties_add_path(props, "template_path", "Template Image", obs.OBS_PATH_FILE, "*.*", None)
    obs.obs_properties_add_text(props, "blur_source", "Blur Source Name", obs.OBS_TEXT_DEFAULT)
    obs.obs_properties_add_float_slider(props, "threshold", "Threshold", 0.50, 0.99, 0.01)
    obs.obs_properties_add_int(props, "monitor", "Monitor Index (1..N)", 1, 8, 1)
    obs.obs_properties_add_int(props, "x", "Region X", 0, 10000, 1)
    obs.obs_properties_add_int(props, "y", "Region Y", 0, 10000, 1)
    obs.obs_properties_add_int(props, "w", "Region Width (0=full monitor)", 0, 10000, 1)
    obs.obs_properties_add_int(props, "h", "Region Height (0=full monitor)", 0, 10000, 1)
    obs.obs_properties_add_int_slider(props, "interval_ms", "Scan Interval (ms)", 30, 1000, 5)
    obs.obs_properties_add_float_slider(props, "downscale", "Downscale", 0.25, 1.0, 0.05)
    obs.obs_properties_add_text(props, "scales", "Template Scales (csv)", obs.OBS_TEXT_DEFAULT)
    obs.obs_properties_add_int_slider(props, "hits_required", "Hits Required", 1, 10, 1)
    obs.obs_properties_add_int_slider(props, "misses_before_hide", "Misses Before Hide", 1, 20, 1)
    obs.obs_properties_add_bool(props, "debug_logs", "Debug Logs")
    return props


def script_update(settings) -> None:
    global _obs_template_path, _obs_blur_source
    global _obs_threshold, _obs_monitor, _obs_region
    global _obs_interval_ms, _obs_downscale, _obs_scales
    global _obs_hits_required, _obs_misses_before_hide
    global _obs_debug_logs
    if obs is None:
        return

    _obs_template_path = obs.obs_data_get_string(settings, "template_path")
    _obs_blur_source = obs.obs_data_get_string(settings, "blur_source")
    _obs_threshold = float(obs.obs_data_get_double(settings, "threshold"))
    _obs_monitor = int(obs.obs_data_get_int(settings, "monitor"))
    _obs_region = (
        int(obs.obs_data_get_int(settings, "x")),
        int(obs.obs_data_get_int(settings, "y")),
        int(obs.obs_data_get_int(settings, "w")),
        int(obs.obs_data_get_int(settings, "h")),
    )
    _obs_interval_ms = int(obs.obs_data_get_int(settings, "interval_ms"))
    _obs_downscale = float(obs.obs_data_get_double(settings, "downscale"))
    _obs_scales = obs.obs_data_get_string(settings, "scales")
    _obs_hits_required = int(obs.obs_data_get_int(settings, "hits_required"))
    _obs_misses_before_hide = int(obs.obs_data_get_int(settings, "misses_before_hide"))
    _obs_debug_logs = bool(obs.obs_data_get_bool(settings, "debug_logs"))

    obs.timer_remove(_obs_tick)
    _obs_rebuild_runtime()
    obs.timer_add(_obs_tick, _obs_interval_ms)


def script_load(settings) -> None:
    if obs is None:
        return
    script_update(settings)


def script_unload() -> None:
    if obs is None:
        return
    obs.timer_remove(_obs_tick)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Template search for Valorant username blurring.")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE_PATH), help="Path to template image.")
    p.add_argument("--threshold", type=float, default=0.83, help="Match threshold (0-1).")
    p.add_argument("--monitor", type=int, default=1, help="Monitor index from mss (1..N).")
    p.add_argument("--x", type=int, default=0, help="Region X (monitor-local).")
    p.add_argument("--y", type=int, default=0, help="Region Y (monitor-local).")
    p.add_argument("--w", type=int, default=0, help="Region width (0 = full monitor).")
    p.add_argument("--h", type=int, default=0, help="Region height (0 = full monitor).")
    p.add_argument("--scales", default="0.95,1.0,1.05", help="CSV scale factors for template.")
    p.add_argument("--downscale", type=float, default=1.0, help="Downscale capture frame for speed.")
    p.add_argument("--fps", type=int, default=10, help="Scan rate for CLI mode.")
    p.add_argument("--debug", action="store_true", help="Show debug window with match box.")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    cli_args = parser.parse_args()
    raise SystemExit(run_cli(cli_args))
