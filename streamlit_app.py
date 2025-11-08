# streamlit_app.py
import io, math
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

# IMPORTANT: make sure this is installed on your host
# pip install streamlit-drawable-canvas==0.9.3
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Osteotomy Live Canvas (robust)", layout="wide")

# ---------------- helpers ----------------
def load_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def to_rgb_pil(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def draw_overlay(img_rgba: Image.Image,
                 poly, prox, dist, cora, hinge) -> Image.Image:
    """Draw persisted geometry on a copy of the given RGBA image."""
    im = img_rgba.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    if len(poly) >= 3:
        draw.polygon(poly, outline=(0,255,255,255), fill=(0,255,255,40))
    if len(prox) == 2:
        draw.line(prox, fill=(66,133,244,255), width=3)
    if len(dist) == 2:
        draw.line(dist, fill=(221,0,221,255), width=3)
    if cora:
        cx, cy = cora
        draw.ellipse([cx-5,cy-5,cx+5,cy+5], outline=(0,200,0,255), width=2)
    if hinge:
        hx, hy = hinge
        draw.ellipse([hx-6,hy-6,hx+6,hy+6], outline=(255,165,0,255), width=3)
        draw.line([(hx-12,hy),(hx+12,hy)], fill=(255,165,0,255), width=1)
        draw.line([(hx,hy-12),(hx,hy+12)], fill=(255,165,0,255), width=1)
    return im

def parse_line(obj) -> list[tuple[float,float]]:
    return [(float(obj.get("x1",0)), float(obj.get("y1",0))),
            (float(obj.get("x2",0)), float(obj.get("y2",0)))]

def parse_poly_points(obj) -> list[tuple[float,float]]:
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

def dist2(a, b):
    dx = a[0]-b[0]; dy = a[1]-b[1]; return dx*dx+dy*dy

def safe_st_canvas(bg_pil: Image.Image, *, width: int, height: int,
                   drawing_mode: str, stroke_color: str, stroke_width: int,
                   key: str):
    """
    Robust wrapper so we work across canvas versions:
      1) Try PIL background_image (preferred by many builds)
      2) Try NumPy HxWx3
      3) Fall back to white background_color
    """
    # 1) PIL
    try:
        return st_canvas(
            background_image=to_rgb_pil(bg_pil),
            width=width, height=height,
            drawing_mode=drawing_mode,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            update_streamlit=True,
            key=key,
        )
    except Exception:
        pass
    # 2) NumPy
    try:
        bg_np = np.array(to_rgb_pil(bg_pil), dtype=np.uint8)
        return st_canvas(
            background_image=bg_np,
            width=width, height=height,
            drawing_mode=drawing_mode,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            update_streamlit=True,
            key=key,
        )
    except Exception:
        pass
    # 3) No background image — use white color (last resort)
    return st_canvas(
        background_image=None,
        background_color="#ffffff",
        width=width, height=height,
        drawing_mode=drawing_mode,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        update_streamlit=True,
        key=key,
    )

# ---------------- state ----------------
ss = st.session_state
defaults = dict(
    poly=[], prox=[], dist=[], cora=None, hinge=None,
    tool="Polygon", preview_w=1100, snap_px=10,
    canvas_nonce=0,
)
for k,v in defaults.items(): ss.setdefault(k, v)

st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
ss.tool = st.sidebar.radio("Tool", ["Polygon","Prox line","Dist line","CORA","HINGE"], index=0)
ss.preview_w = st.sidebar.slider("Preview width", 600, 1800, ss.preview_w, 50)
ss.snap_px   = st.sidebar.slider("Polygon snap distance (px)", 4, 20, int(ss.snap_px), 1)

c1,c2,c3,c4 = st.sidebar.columns(4)
if c1.button("Reset poly"):   ss.poly.clear(); ss.canvas_nonce += 1
if c2.button("Reset lines"):  ss.prox.clear(); ss.dist.clear(); ss.canvas_nonce += 1
if c3.button("Clear points"): ss.cora=None; ss.hinge=None; ss.canvas_nonce += 1
if c4.button("Clear canvas"): ss.canvas_nonce += 1

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# ---------- prepare preview base ----------
src_rgba = load_image(up.getvalue())              # full-res
W,H = src_rgba.size
scale = min(ss.preview_w/float(W), 1.0)
cw,ch = int(round(W*scale)), int(round(H*scale))
base_rgba = src_rgba.resize((cw,ch), Image.NEAREST)

# draw persisted geometry on the preview base (so it shows under live strokes)
bg_preview = draw_overlay(base_rgba, ss.poly, ss.prox, ss.dist, ss.cora, ss.hinge)

# ---------- choose canvas mode ----------
drawing_mode = "transform"
stroke_color = "#00FFFF"
stroke_width = 2
if ss.tool == "Polygon":
    drawing_mode = "polyline"      # we auto-snap-close
    stroke_color = "#00FFFF"; stroke_width = 2
elif ss.tool == "Prox line":
    drawing_mode = "line"; stroke_color = "#4285F4"; stroke_width = 3
elif ss.tool == "Dist line":
    drawing_mode = "line"; stroke_color = "#DD00DD"; stroke_width = 3
# CORA/HINGE use inputs below; keep canvas active with "transform"

canvas_key = f"canvas-{ss.canvas_nonce}-{ss.tool}"
result = safe_st_canvas(bg_preview, width=cw, height=ch,
                        drawing_mode=drawing_mode,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        key=canvas_key)

# ---------- handle canvas output ----------
if result.json_data:
    objs = result.json_data.get("objects", [])
    for obj in reversed(objs):
        typ = obj.get("type","")
        if ss.tool == "Polygon" and typ in ("polyline","path","polygon"):
            pts = parse_poly_points(obj)
            if len(pts) >= 2:
                p0 = pts[0]; plast = pts[-1]
                if dist2(plast, p0) <= (ss.snap_px*ss.snap_px) and len(pts) >= 3:
                    closed = pts[:-1] + [p0]
                    ss.poly = [(x/scale, y/scale) for (x,y) in closed]
                    ss.canvas_nonce += 1
                    st.experimental_rerun()
            break
        if ss.tool in ("Prox line","Dist line") and typ == "line":
            seg = parse_line(obj)
            seg_full = [(x/scale, y/scale) for (x,y) in seg]
            if ss.tool == "Prox line": ss.prox = seg_full
            else: ss.dist = seg_full
            break

# ---------- CORA/HINGE quick inputs (simple & robust) ----------
st.sidebar.markdown("---")
colA,colB = st.sidebar.columns(2)
with colA:
    cx = st.number_input("CORA x", value=float(ss.cora[0]) if ss.cora else 0.0, step=1.0)
    hx = st.number_input("HINGE x", value=float(ss.hinge[0]) if ss.hinge else 0.0, step=1.0)
with colB:
    cy = st.number_input("CORA y", value=float(ss.cora[1]) if ss.cora else 0.0, step=1.0)
    hy = st.number_input("HINGE y", value=float(ss.hinge[1]) if ss.hinge else 0.0, step=1.0)
if st.sidebar.button("Set CORA/HINGE"):
    ss.cora  = (cx, cy)
    ss.hinge = (hx, hy)
    ss.canvas_nonce += 1
    st.experimental_rerun()

# ---------- show persisted overlay preview ----------
st.markdown("### Stored overlay (scaled preview)")
preview = draw_overlay(src_rgba, ss.poly, ss.prox, ss.dist, ss.cora, ss.hinge)
st.image(preview.resize((cw,ch), Image.NEAREST))
