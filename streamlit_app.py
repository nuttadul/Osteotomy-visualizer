import io, math
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
from streamlit_drawable_canvas import st_canvas    # pip install streamlit-drawable-canvas

st.set_page_config(page_title="Osteotomy Live Canvas", layout="wide")

# ---------- helpers ----------
def load_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    """Return HxWx3 uint8 RGB array (what st_canvas needs)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.uint8)

def draw_overlay_on(pil_img: Image.Image, poly, prox, dist, cora, hinge) -> Image.Image:
    """Draw persisted overlay (poly, lines, points) on top of an RGBA copy."""
    im = pil_img.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    # poly
    if len(poly) >= 3:
        draw.polygon(poly, outline=(0, 255, 255, 255), fill=(0, 255, 255, 40))
    # prox line
    if len(prox) == 2:
        draw.line(prox, fill=(66, 133, 244, 255), width=3)
    # dist line
    if len(dist) == 2:
        draw.line(dist, fill=(221, 0, 221, 255), width=3)
    # points
    if cora:
        cx, cy = cora; draw.ellipse([cx-5, cy-5, cx+5, cy+5], outline=(0,200,0,255), width=2)
    if hinge:
        hx, hy = hinge
        draw.ellipse([hx-6, hy-6, hx+6, hy+6], outline=(255,165,0,255), width=3)
        draw.line([(hx-12, hy), (hx+12, hy)], fill=(255,165,0,255), width=1)
        draw.line([(hx, hy-12), (hx, hy+12)], fill=(255,165,0,255), width=1)
    return im

def parse_line(obj) -> List[Tuple[float,float]]:
    return [(float(obj.get("x1",0)), float(obj.get("y1",0))),
            (float(obj.get("x2",0)), float(obj.get("y2",0)))]

def parse_poly_points(obj) -> List[Tuple[float,float]]:
    """Accepts 'points' (preferred) or 'path' from canvas polyline object."""
    if "points" in obj and isinstance(obj["points"], list):
        left = float(obj.get("left", 0.0)); top = float(obj.get("top", 0.0))
        sx = float(obj.get("scaleX", 1.0)); sy = float(obj.get("scaleY", 1.0))
        pts = []
        for p in obj["points"]:
            px = left + sx * float(p.get("x", 0.0))
            py = top + sy * float(p.get("y", 0.0))
            pts.append((px, py))
        return pts
    # very old builds sometimes report a 'path'
    if "path" in obj and isinstance(obj["path"], list):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M","L"):
                pts.append((float(cmd[1]), float(cmd[2])))
        return pts
    return []

def dist2(a, b):  # squared distance
    dx = a[0]-b[0]; dy = a[1]-b[1]; return dx*dx + dy*dy

# ---------- state ----------
ss = st.session_state
defaults = dict(
    poly=[],             # closed polygon [(x,y), ...]
    prox=[],             # 2 points
    dist=[],             # 2 points
    cora=None,
    hinge=None,
    tool="Polygon",
    snap_px=10.0,
    canvas_key_nonce=0,  # bump to clear the canvas
    preview_w=1100,
)
for k,v in defaults.items(): ss.setdefault(k, v)

st.sidebar.header("Upload image")
uploaded = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool = st.sidebar.radio("Tool", ["Polygon", "Prox line", "Dist line", "CORA", "HINGE"], index=0)

# Controls that affect preview size only
ss.preview_w = st.sidebar.slider("Preview width", 600, 1800, ss.preview_w, 50)
ss.snap_px    = st.sidebar.slider("Polygon snap distance (px)", 4, 20, int(ss.snap_px), 1)

c1, c2, c3, c4 = st.sidebar.columns(4)
if c1.button("Reset poly"):     ss.poly.clear()
if c2.button("Reset lines"):    ss.prox.clear(); ss.dist.clear()
if c3.button("Clear points"):   ss.cora=None; ss.hinge=None
if c4.button("Clear canvas"):   ss.canvas_key_nonce += 1

if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

# ---------- image prep ----------
src_rgba = load_image(uploaded.getvalue())                  # full-res RGBA
# scale preview canvas (keep ≤ preview_w)
W, H = src_rgba.size
scale = min(ss.preview_w / float(W), 1.0)
cw, ch = int(round(W*scale)), int(round(H*scale))

# rasterize persisted overlay (poly/lines/points) on a copy for user feedback (background layer)
overlay_rgba = draw_overlay_on(src_rgba.resize((cw,ch), Image.NEAREST), ss.poly, ss.prox, ss.dist, ss.cora, ss.hinge)
bg_np = pil_to_rgb_np(overlay_rgba)                         # HxWx3 RGB for canvas

# ---------- choose drawing mode ----------
drawing_mode = None
stroke_color = "#00FFFF"
stroke_width = 2
if ss.tool == "Polygon":
    drawing_mode = "polyline"      # use polyline + auto-snap close
    stroke_color = "#00FFFF"
    stroke_width = 2
elif ss.tool == "Prox line":
    drawing_mode = "line"
    stroke_color = "#4285F4"
    stroke_width = 3
elif ss.tool == "Dist line":
    drawing_mode = "line"
    stroke_color = "#DD00DD"
    stroke_width = 3
else:
    drawing_mode = "transform"     # no stroke; just to keep canvas active

# ---------- canvas ----------
canvas_key = f"canvas-{ss.canvas_key_nonce}-{ss.tool}"
result = st_canvas(
    background_image=bg_np,            # this prevents white canvas
    width=cw,
    height=ch,
    drawing_mode=drawing_mode,
    stroke_color=stroke_color,
    stroke_width=stroke_width,
    update_streamlit=True,
    key=canvas_key,
)

# ---------- read back objects ----------
if result.json_data:
    objs = result.json_data.get("objects", [])
    # look at most recent first
    for obj in reversed(objs):
        typ = obj.get("type", "")
        # polygon via polyline + auto-snap
        if ss.tool == "Polygon" and typ in ("polyline", "path", "polygon"):
            pts = parse_poly_points(obj)
            if len(pts) >= 2:
                # live preview—if user came close to first point, snap and close
                p0 = pts[0]; plast = pts[-1]
                if dist2(plast, p0) <= (ss.snap_px*ss.snap_px) and len(pts) >= 3:
                    closed = pts[:-1] + [p0]    # close the loop
                    # store in **original** pixel space, not scaled
                    closed_full = [(x/scale, y/scale) for (x,y) in closed]
                    ss.poly = closed_full
                    ss.canvas_key_nonce += 1     # clear strokes after closing
                    st.experimental_rerun()
                else:
                    # not closed yet—preview only (no store)
                    pass
            break

        # prox/dist line
        if ss.tool in ("Prox line", "Dist line") and typ == "line":
            line = parse_line(obj)
            line_full = [(x/scale, y/scale) for (x,y) in line]
            if ss.tool == "Prox line":
                ss.prox = line_full
            else:
                ss.dist = line_full
            break

# ---------- CORA/HINGE clicks (simple text inputs as a quick workaround) ----------
st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    cx = st.number_input("CORA x", value=float(ss.cora[0]) if ss.cora else 0.0, step=1.0)
    hx = st.number_input("HINGE x", value=float(ss.hinge[0]) if ss.hinge else 0.0, step=1.0)
with col_b:
    cy = st.number_input("CORA y", value=float(ss.cora[1]) if ss.cora else 0.0, step=1.0)
    hy = st.number_input("HINGE y", value=float(ss.hinge[1]) if ss.hinge else 0.0, step=1.0)
if st.sidebar.button("Set CORA/HINGE"):
    ss.cora  = (cx, cy)
    ss.hinge = (hx, hy)

# ---------- preview of stored geometry (exact, full-res scaled down) ----------
st.markdown("### Stored overlay preview (persisted)")
preview = draw_overlay_on(src_rgba, ss.poly, ss.prox, ss.dist, ss.cora, ss.hinge)
st.image(preview.resize((cw, ch), Image.NEAREST), use_column_width=False)
