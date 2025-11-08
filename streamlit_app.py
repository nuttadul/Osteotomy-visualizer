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

# --------- image + scale ----------
src_rgba = load_rgba(up.getvalue())
W,H = src_rgba.size
scale = min(ss.preview_w/float(W), 1.0)
cw,ch = int(round(W*scale)), int(round(H*scale))
base_rgba = src_rgba.resize((cw,ch), Image.NEAREST)

# Fabric background image (locked object via initial_drawing)
bg_dataurl = pil_to_dataurl(base_rgba.convert("RGB"))
initial_drawing = {
    "version": "5.2.4",
    "objects": [{
        "type": "image",
        "left": 0, "top": 0,
        "width": cw, "height": ch,
        "scaleX": 1, "scaleY": 1,
        "angle": 0,
        "flipX": False, "flipY": False,
        "opacity": 1,
        "src": bg_dataurl,
        "selectable": False,
        "evented": False,
        "hasControls": False,
        "hasBorders": False,
        "objectCaching": False
    }]
}

# --------- UI layout ----------
left, right = st.columns([1.1, 1])
left.subheader("Live drawing (with real image background)")
right.subheader("Preview (persisted shapes; click here for CORA/HINGE)")

# ---------- Canvas (works even if background_image is ignored) ----------
if ss.tool == "Polygon":
    drawing_mode = "polyline"; stroke_color = "#00FFFF"; width_px = 2
elif ss.tool == "Prox line":
    drawing_mode = "line";     stroke_color = "#4285F4"; width_px = 3
elif ss.tool == "Dist line":
    drawing_mode = "line";     stroke_color = "#DD00DD"; width_px = 3
else:
    drawing_mode = "transform"   # idle
    stroke_color = "#00FFFF"; width_px = 1

with left:
    result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=width_px,
        stroke_color=stroke_color,
        background_color=None,                 # we use locked image object instead
        initial_drawing=json.dumps(initial_drawing),
        update_streamlit=True,
        width=cw, height=ch,
        drawing_mode=drawing_mode,
        key=f"canvas-{ss.nonce}-{ss.tool}",
        display_toolbar=True,
    )

# Read back the canvas only for drawing tools
if result.json_data and ss.tool in ("Polygon","Prox line","Dist line"):
    objs = result.json_data.get("objects", [])
    # We drew the image as the 1st object; usable shapes are later objects.
    # Scan from the end to get the most recent line/polyline.
    for obj in reversed(objs):
        typ = obj.get("type","")
        if typ == "image":
            continue
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

# ---------- Preview + click-to-set CORA / HINGE ----------
with right:
    preview_rgba = draw_overlay(base_rgba, ss.poly, ss.prox, ss.dist, ss.cora, ss.hinge)
    if ss.tool in ("CORA","HINGE"):
        st.caption("Click anywhere on the image to set the point.")
        res = streamlit_image_coordinates(preview_rgba, width=cw, key=f"clicks-{ss.nonce}")
        if res and "x" in res and "y" in res:
            pt = (float(res["x"])/scale, float(res["y"])/scale)
            if ss.tool == "CORA":  ss.cora  = pt
            if ss.tool == "HINGE": ss.hinge = pt
            ss.nonce += 1
            st.experimental_rerun()
    else:
        st.image(preview_rgba, width=cw)
