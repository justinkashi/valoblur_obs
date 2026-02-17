# valoblur_obs

OBS helper script to detect your Valorant username on screen and toggle a blur source.

## Project Layout

- `scripts/auto_blur.py`: Template matcher + OBS script hooks.
- `assets/template.png`: Cropped screenshot of your username (add this file).
- `requirements.txt`: Python dependencies.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Quick Local Test (CLI Mode)

Use this to tune threshold/region before OBS:

```bash
python3 scripts/auto_blur.py \
  --template assets/template.png \
  --threshold 0.83 \
  --monitor 1 \
  --x 0 --y 0 --w 0 --h 0 \
  --fps 10 \
  --debug
```

Notes:
- Press `q` to close debug preview.
- Set `--w`/`--h` to a smaller HUD region for lower CPU usage.
- Try `--scales 0.92,0.96,1.0,1.04` if your UI scale varies.

## OBS Setup

1. In OBS, open `Tools > Scripts`.
2. Add `scripts/auto_blur.py`.
3. Set:
   - `Template Image`: path to `assets/template.png`
   - `Blur Source Name`: source to show/hide (e.g. blur overlay)
   - `Threshold`: start at `0.83`
   - `Monitor Index` and region values
4. Tune:
   - `Scan Interval`: 80-150 ms is usually enough.
   - `Downscale`: `0.5`-`1.0` (lower is faster).
   - `Hits Required` / `Misses Before Hide` to reduce flicker.

If blur does not trigger, lower threshold gradually (`0.80`, `0.78`) or recrop template tighter.
