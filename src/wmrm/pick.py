"""Draw the box with a mouse instead of typing coordinates.

NOT WIRED INTO THE CLI. Groundwork for the future UI phase, parked deliberately:
drawing needs a browser, and the tool currently runs in headless containers where
opening an HTML file is not practical. The CLI stays "detect -> preview PNG ->
confirm or pass coordinates".

When the UI does get built, this is the piece to start from. It writes one
self-contained HTML file with a frame of the video embedded, you drag a rectangle
over the watermark, and it prints the ready-to-paste commands. It also seeds the
rectangle from a detection guess when one is available, so the usual interaction
is a nudge rather than a fresh drag.

The file is local and stays local: the frame is embedded as a data URI, nothing is
uploaded and nothing is fetched.

    from wmrm.pick import write_picker
    write_picker(Path("in.mp4"), Path("out.html"), box=(1554, 44, 284, 62))
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from .probe import probe, require_tools


class PickError(RuntimeError):
    pass


_HTML = """<!doctype html>
<meta charset="utf-8">
<title>wmrm — pick the watermark box</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 16px;
          background: Canvas; color: CanvasText; }}
  h1 {{ font-size: 15px; margin: 0 0 4px; }}
  p.hint {{ margin: 0 0 12px; opacity: .7; }}
  #wrap {{ position: relative; display: inline-block; max-width: 100%;
           cursor: crosshair; }}
  #frame {{ display: block; max-width: 100%; height: auto; }}
  #box {{ position: absolute; border: 2px solid #ff2d2d; background: #ff2d2d18;
          pointer-events: none; display: none; }}
  #bar {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 14px;
          align-items: flex-end; }}
  label {{ display: block; font-size: 12px; opacity: .7; }}
  input {{ font: inherit; width: 7em; padding: 4px 6px; }}
  #cmd {{ margin-top: 14px; padding: 10px 12px; border-radius: 6px;
          background: color-mix(in srgb, CanvasText 8%, Canvas);
          font-family: ui-monospace, monospace; overflow-x: auto;
          white-space: pre; }}
  button {{ font: inherit; padding: 5px 10px; }}
  #zoomwrap {{ margin-top: 12px; }}
  #zoom {{ border: 1px solid color-mix(in srgb, CanvasText 25%, Canvas);
           image-rendering: pixelated; }}
</style>

<h1>{name} &middot; {w}&times;{h}</h1>
<p class="hint">Drag a rectangle over the watermark. Nudge with the number
fields or arrow keys, then copy the command at the bottom.</p>

<div id="wrap">
  <img id="frame" src="data:image/png;base64,{b64}" alt="frame">
  <div id="box"></div>
</div>

<div id="bar">
  <div><label for="ix">x</label><input id="ix" type="number" value="{bx}"></div>
  <div><label for="iy">y</label><input id="iy" type="number" value="{by}"></div>
  <div><label for="iw">w</label><input id="iw" type="number" value="{bw}"></div>
  <div><label for="ih">h</label><input id="ih" type="number" value="{bh}"></div>
  <button id="grow" title="grow 2px each side">grow</button>
  <button id="shrink">shrink</button>
  <button id="copy">copy command</button>
</div>

<div id="zoomwrap"><canvas id="zoom" width="480" height="150"></canvas></div>
<div id="cmd"></div>

<script>
const SRC_W = {w}, SRC_H = {h}, NAME = {name_json};
const img = document.getElementById('frame'), wrap = document.getElementById('wrap');
const boxEl = document.getElementById('box'), cmd = document.getElementById('cmd');
const zoom = document.getElementById('zoom'), zctx = zoom.getContext('2d');
const F = {{ x: document.getElementById('ix'), y: document.getElementById('iy'),
            w: document.getElementById('iw'), h: document.getElementById('ih') }};
let box = {{ x: {bx}, y: {by}, w: {bw}, h: {bh} }};

// The image is CSS-scaled to fit the window, so every screen coordinate has to be
// converted through the natural/displayed ratio to land on real source pixels.
const ratio = () => SRC_W / img.clientWidth;

function clampBox() {{
  box.w = Math.max(1, Math.round(box.w));
  box.h = Math.max(1, Math.round(box.h));
  box.x = Math.min(Math.max(0, Math.round(box.x)), SRC_W - 1);
  box.y = Math.min(Math.max(0, Math.round(box.y)), SRC_H - 1);
  box.w = Math.min(box.w, SRC_W - box.x);
  box.h = Math.min(box.h, SRC_H - box.y);
}}

function render() {{
  clampBox();
  const r = 1 / ratio();
  Object.assign(boxEl.style, {{
    display: 'block', left: box.x * r + 'px', top: box.y * r + 'px',
    width: box.w * r + 'px', height: box.h * r + 'px',
  }});
  for (const k in F) if (document.activeElement !== F[k]) F[k].value = box[k];
  cmd.textContent =
    `wmrm run ${{NAME}} --box ${{box.x}},${{box.y}},${{box.w}},${{box.h}} --preview-only\\n` +
    `wmrm detect ${{NAME}} --box ${{box.x}},${{box.y}},${{box.w}},${{box.h}} --preset mark.json\\n` +
    `wmrm run ${{NAME}} --preset mark.json`;
  drawZoom();
}}

function drawZoom() {{
  const pad = 24, sw = box.w + pad * 2, sh = box.h + pad * 2;
  const s = Math.min(zoom.width / sw, zoom.height / sh);
  zctx.clearRect(0, 0, zoom.width, zoom.height);
  zctx.imageSmoothingEnabled = false;
  const dw = sw * s, dh = sh * s, ox = (zoom.width - dw) / 2, oy = (zoom.height - dh) / 2;
  zctx.drawImage(img, box.x - pad, box.y - pad, sw, sh, ox, oy, dw, dh);
  zctx.strokeStyle = '#ff2d2d'; zctx.lineWidth = 1;
  zctx.strokeRect(ox + pad * s, oy + pad * s, box.w * s, box.h * s);
}}

let drag = null;
wrap.addEventListener('pointerdown', e => {{
  const b = img.getBoundingClientRect(), k = ratio();
  drag = {{ x: (e.clientX - b.left) * k, y: (e.clientY - b.top) * k }};
  wrap.setPointerCapture(e.pointerId);
}});
wrap.addEventListener('pointermove', e => {{
  if (!drag) return;
  const b = img.getBoundingClientRect(), k = ratio();
  const cx = (e.clientX - b.left) * k, cy = (e.clientY - b.top) * k;
  box = {{ x: Math.min(drag.x, cx), y: Math.min(drag.y, cy),
          w: Math.abs(cx - drag.x), h: Math.abs(cy - drag.y) }};
  render();
}});
wrap.addEventListener('pointerup', () => {{ drag = null; }});

for (const k in F) F[k].addEventListener('input', () => {{
  const v = parseInt(F[k].value, 10);
  if (!Number.isNaN(v)) {{ box[k] = v; render(); }}
}});
document.getElementById('grow').onclick = () => {{
  box.x -= 2; box.y -= 2; box.w += 4; box.h += 4; render();
}};
document.getElementById('shrink').onclick = () => {{
  box.x += 2; box.y += 2; box.w -= 4; box.h -= 4; render();
}};
document.getElementById('copy').onclick = async () => {{
  try {{ await navigator.clipboard.writeText(cmd.textContent); }} catch {{}}
}};
addEventListener('keydown', e => {{
  const step = e.shiftKey ? 10 : 1;
  const moves = {{ ArrowLeft: ['x', -step], ArrowRight: ['x', step],
                  ArrowUp: ['y', -step], ArrowDown: ['y', step] }};
  if (moves[e.key] && document.activeElement.tagName !== 'INPUT') {{
    const [k, d] = moves[e.key]; box[k] += d; render(); e.preventDefault();
  }}
}});
addEventListener('resize', render);
img.complete ? render() : img.addEventListener('load', render);
</script>
"""


def write_picker(src: Path, out_html: Path, *, at: float | None = None,
                 box: tuple[int, int, int, int] | None = None) -> Path:
    ffmpeg, _ = require_tools()
    info = probe(src)
    t = at if at is not None else max(0.0, info.duration / 2)

    res = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(src),
         "-frames:v", "1", "-f", "image2", "-c:v", "png", "-"],
        capture_output=True,
    )
    if res.returncode != 0 or not res.stdout:
        raise PickError(f"could not extract a frame from {src}:\n"
                        f"{res.stderr.decode('utf-8', 'replace')[:400]}")

    bx, by, bw, bh = box if box else (
        int(info.width * 0.80), int(info.height * 0.02),
        int(info.width * 0.15), int(info.height * 0.06),
    )
    import json

    html = _HTML.format(
        name=src.name, name_json=json.dumps(src.name),
        w=info.width, h=info.height,
        b64=base64.b64encode(res.stdout).decode("ascii"),
        bx=bx, by=by, bw=bw, bh=bh,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html
