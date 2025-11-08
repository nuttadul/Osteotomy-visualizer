# streamlit_app.py
import io, math
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st

# Pin a compatible version in requirements.txt:
# streamlit
# Pillow
# numpy
# streamlit-drawable-canvas==0.9.3

from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Osteotomy – robust canvas", layout="wide")

# ---------------- utils ----------------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def to_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB") if im.mode != "RGB" else im

def draw_persisted(base_rgba: Image.Image,
                   poly, prox, dist, cora, hinge) -> Image.Image:
    out = base_rgba.copy()
    d = ImageDraw.Draw(out, "RGBA")
    if len(poly) >= 3:
        d.polygon(poly, outline=(0,255,255,255), fill=(0,255,255,40))
    if len(prox) == 2:
        d.line(prox, fill=(66,133,244,255), width=3)
    if len(dist) == 2:
        d.line(dist, fill=(221,0,221,255), width=3)
    if cora:
        x,y = cora; d.ellipse([x-5,y-5,x+5,y+5], outline=(0,200,0,255), width=2)
    if hinge:
        x,y = hinge
        d.ellipse([x-6,y-6,x+6,y+6], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)
    return out

def parse_line(obj):
    return [(float(obj.get("x1",0)), float(obj.get("y1",0))),
            (float(obj.get("x2",0)), float(obj.get("y2",0)))]

def parse_points(obj):
    # Prefer 'points'
    if "points" in obj and isinstance(obj["points"], list):
        left = float(obj.get("left",0)); top = float(obj.get("top",0))
        sx = float(obj.get("scaleX",1)); sy = float(obj.get("scaleY",1))
        pts = []
        for p in obj["points"]:
            px = left + sx * float(p.get("x",0))
            py = top  + sy * float(p.get("y",0))
            pts.append((px,py))
        return pts
    # Fallback 'path'
    if "path" in obj and isinstance(obj["path"], list):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M","L"):
                pts.append((float(cmd[1]), float(cmd[2])))
        return pts
    return []

def d2(a,b): x=a[0]-b[0]; y=a[1]-b[1]; return x*x+y*y

# ---------------- state ----------------
ss = st.session_state
defaults = dict(poly=[], prox=[], dist=[], cora=None, hinge=None,
                tool="Polygon", preview_w=1100, snap_px=10, nonce=0)
for k,v in defaults.items(): ss.setdefault(k,v)

st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
ss.tool      = st.sidebar.radio("Tool", ["Polygon","Prox line","Dist line"], index=0)
ss.preview_w = st.sidebar.slider("Preview width", 600, 1800, ss.preview_w, 50)
ss.snap_px   = st.sidebar.slider("Polygon snap distance (px)", 4, 20, int(ss.snap_px), 1)

c1,c2,c3 = st.sidebar.columns(3)
if c1.button("Reset poly"):  ss.poly.clear(); ss.nonce += 1
if c2.button("Reset lines"): ss.prox.clear(); ss.dist.clear(); ss.nonce += 1
if c3.button("Clear all"):  ss.poly.clear(); ss.prox.clear(); ss.dist.clear(); ss.cora=None; ss.hinge=None; ss.nonce += 1

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# ------------- prepare base preview -------------
src_rgba = load_rgba(up.getvalue())
W,H = src_rgba.size
scale = min(ss.preview_w/float(W), 1.0)
cw,ch = int(round(W*scale)), int(round(H*scale))
base_rgba = src_rgba.resize((cw,ch), Image.NEAREST)

# draw persisted geometry into the background image itself
bg_rgba = draw_persisted(base_rgba, ss.poly, ss.prox, ss.dist, ss.cora, ss.hinge)
bg_rgb  = to_rgb(bg_rgba)              # <-- many canvas builds need plain RGB

# ----------- choose mode & stroke ------------
if ss.tool == "Polygon":
    drawing_mode = "polyline"          # we auto-close by snapping
    stroke_color = "#00FFFF"; stroke_width = 2
elif ss.tool == "Prox line":
    drawing_mode = "line"; stroke_color = "#4285F4"; stroke_width = 3
else:
    drawing_mode = "line"; stroke_color = "#DD00DD"; stroke_width = 3

# ----------- SAFEST call form ---------------
# Pass ALL canonical parameters, always.
def call_canvas_with(bg):
    return st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#ffffff",        # ignored when background_image is set
        background_image=bg,               # PIL RGB or NumPy HxWx3
        update_streamlit=True,
        width=cw,
        height=ch,
        drawing_mode=drawing_mode,
        key=f"canvas-{ss.nonce}-{ss.tool}",
    )

# 1) PIL RGB (preferred by many builds)
try:
    result = call_canvas_with(bg_rgb)
except Exception:
    # 2) NumPy fallback
    try:
        result = call_canvas_with(np.array(bg_rgb, dtype=np.uint8))
    except Exception:
        # 3) Last resort: no image → at least tool won’t crash
        result = call_canvas_with(None)

# ----------- read canvas output --------------
if result.json_data:
    objs = result.json_data.get("objects", [])
    for obj in reversed(objs):
        typ = obj.get("type","")
        if ss.tool == "Polygon" and typ in ("polyline","path","polygon"):
            pts = parse_points(obj)
            if len(pts) >= 2:
                p0, plast = pts[0], pts[-1]
                if d2(plast, p0) <= (ss.snap_px*ss.snap_px) and len(pts) >= 3:
                    closed = pts[:-1] + [p0]
                    ss.poly = [(x/scale, y/scale) for (x,y) in closed]
                    ss.nonce += 1
                    st.experimental_rerun()
            break
        if typ == "line" and ss.tool in ("Prox line","Dist line"):
            seg = parse_line(obj)
            seg_full = [(x/scale, y/scale) for (x,y) in seg]
            if ss.tool == "Prox line": ss.prox = seg_full
            else: ss.dist = seg_full
            break

# ------------- show persisted overlay -------------
st.markdown("### Preview (stored overlay)")
st.image(draw_persisted(src_rgba, ss.poly, ss.prox, ss.dist, ss.cora, ss.hinge).resize((cw,ch), Image.NEAREST))
