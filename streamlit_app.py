# app.py
import io, base64, json, math
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – canvas with real background", layout="wide")

# ---------------- helpers ----------------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def pil_to_dataurl(img: Image.Image, fmt="PNG") -> str:
    """Encode a PIL image to data URL for Fabric 'image' object."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def draw_overlay(base_rgba: Image.Image,
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
    # Prefer Fabric.js 'points'
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
for k,v in dict(poly=[], prox=[], dist=[], cora=None, hinge=None,
                tool="Polygon", preview_w=1100, snap_px=12, nonce=0).items():
    ss.setdefault(k,v)

st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool      = st.sidebar.radio("Tool", ["Polygon","Prox line","Dist line","CORA","HINGE"], index=0)
ss.preview_w = st.sidebar.slider("Preview width", 600, 1800, ss.preview_w, 50)
ss.snap_px   = st.sidebar.slider("Polygon snap distance (px)", 4, 20, int(ss.snap_px), 1)

c1,c2,c3 = st.sidebar.columns(3)
if c1.button("Reset poly"):  ss.poly.clear(); ss.nonce += 1
if c2.button("Reset lines"): ss.prox.clear(); ss.dist.clear(); ss.nonce += 1
if c3.button("Clear all"):  ss.poly.clear(); ss.prox.clear(); ss.dist.clear(); ss.cora=None; ss.hinge=None; ss.nonce += 1

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# ---------- sizing: one scale for both canvas and preview ----------
# --------- image + scale ----------
src_rgba = load_rgba(up.getvalue())
W, H = src_rgba.size
scale = min(ss.preview_w / float(W), 1.0)
cw, ch = int(round(W * scale)), int(round(H * scale))
base_rgba = src_rgba.resize((cw, ch), Image.NEAREST)
cw = min(ss.dispw, origW)                 # canvas width
scale = cw / float(origW)                 # single source of truth
ch = int(round(origH * scale))            # canvas height

def c2o(pt):  # canvas -> original
    return (pt[0] / scale, pt[1] / scale)

def o2c(pt):  # original -> canvas
    return (pt[0] * scale, pt[1] * scale)

# resized image used BOTH by live canvas background AND the preview
disp_img = img.resize((cw, ch), Image.NEAREST)

# ---------- build initial_drawing with a locked image background ----------
# (works across canvas versions without background_image bugs)
import base64, io as _io
buf = _io.BytesIO()
disp_img.convert("RGB").save(buf, format="PNG")
data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

initial_drawing = {
    "version": "5.3.0",
    "objects": [{
        "type": "image",
        "src": data_url,
        "left": 0, "top": 0,
        "width": cw, "height": ch,
        "scaleX": 1, "scaleY": 1,
        "selectable": False,
        "evented": False
    }]
}

# ---------- helpers to read canvas JSON ----------
def _last_line(objs):
    for obj in reversed(objs):
        if obj.get("type") == "line":
            return [(obj["x1"], obj["y1"]), (obj["x2"], obj["y2"])]
    return None

def _poly_points_from_fabric(obj):
    """Handle both 'polyline' and 'path'."""
    if obj.get("type") == "polyline" and "points" in obj:
        left = obj.get("left", 0); top = obj.get("top", 0)
        sx = obj.get("scaleX", 1.0); sy = obj.get("scaleY", 1.0)
        pts = [(left + sx * p["x"], top + sy * p["y"]) for p in obj["points"]]
        return pts
    if obj.get("type") == "path" and "path" in obj:
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M", "L"):
                pts.append((cmd[1], cmd[2]))
        return pts
    return []

def _last_polyline(objs):
    for obj in reversed(objs):
        t = obj.get("type")
        if t in ("polyline", "path"):
            return _poly_points_from_fabric(obj)
    return None

# setup polygon state if missing
ss.setdefault("poly_open", True)  # True while user is still placing points

# ---------- live canvas ----------
from streamlit_drawable_canvas import st_canvas
result = st_canvas(
    stroke_color="#00A2FF" if ss.tool_prev in ("Prox line", "Dist line") else "#00FFFF",
    stroke_width=3,
    fill_color="rgba(0,0,0,0)",
    background_color=None,
    initial_drawing=initial_drawing,   # IMPORTANT: pass dict (not json.dumps)
    update_streamlit=True,
    width=cw, height=ch,
    drawing_mode=(
        "line" if tool in ("Prox line", "Dist line")
        else ("polyline" if tool == "Polygon" and ss.poly_open else "transform")
    ),
    key=f"live-{ss.click_nonce}-{tool}",
    display_toolbar=True,
)

# ---------- capture shapes from canvas and persist in ORIGINAL pixels ----------
if result.json_data:
    objs = result.json_data.get("objects", [])

    # lines
    line = _last_line(objs)
    if line and tool in ("Prox line", "Dist line"):
        p0, p1 = line
        if tool == "Prox line":
            ss.prox = [c2o(p0), c2o(p1)]
        else:
            ss.dist = [c2o(p0), c2o(p1)]

    # polygon with snap-to-close
    if tool == "Polygon" and ss.poly_open:
        pts_canvas = _last_polyline(objs)
        if pts_canvas and len(pts_canvas) >= 2:
            # snap close when last point is near first
            snap_px = 10   # adjust feel
            p0 = pts_canvas[0]; pend = pts_canvas[-1]
            if ((p0[0]-pend[0])**2 + (p0[1]-pend[1])**2) ** 0.5 <= snap_px:
                pts_canvas[-1] = p0       # close loop exactly
                ss.poly = [c2o(p) for p in pts_canvas]
                ss.poly_open = False      # finished placing
            else:
                # live feedback but not persisted yet
                ss.poly = [c2o(p) for p in pts_canvas]

# allow “Reset poly” button to re-open placement
if c2.button("Reset poly"):
    ss.poly = []
    ss.poly_open = True

# ---------- right-side PREVIEW (same (cw,ch) size = same coordinates) ----------
preview = disp_img.copy()
draw = ImageDraw.Draw(preview)

# draw persisted polygon
if len(ss.poly) >= 3:
    pts_c = [o2c(p) for p in ss.poly]
    draw.polygon(pts_c, outline=(0,255,255,255), fill=(0,255,255,50), width=2)

# persisted lines
if len(ss.prox) == 2:
    draw.line([o2c(ss.prox[0]), o2c(ss.prox[1])], fill=(66,133,244,255), width=3)
if len(ss.dist) == 2:
    draw.line([o2c(ss.dist[0]), o2c(ss.dist[1])], fill=(221,0,221,255), width=3)

st.image(preview, caption="Preview (persisted shapes; click here for CORA/HINGE)", use_column_width=False)
