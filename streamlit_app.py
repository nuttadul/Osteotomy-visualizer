
import io
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_drawable_canvas_fork import st_canvas
import pandas as pd

st.set_page_config(page_title="Bone Tool (faithful web clone)", layout="wide")

def decode_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def polygon_mask(size, pts):
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def apply_affine(img, dx, dy, rot_deg, center):
    rot = img.rotate(rot_deg, resample=Image.BICUBIC, center=center, expand=False)
    out = Image.new("RGBA", img.size, (0,0,0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def centroid(pts):
    if len(pts) < 3: return None
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    a = 0.0; cx = 0.0; cy = 0.0
    for i in range(len(pts)):
        j = (i+1) % len(pts)
        cross = x[i]*y[j] - x[j]*y[i]
        a += cross
        cx += (x[i] + x[j]) * cross
        cy += (y[i] + y[j]) * cross
    a *= 0.5
    if abs(a) < 1e-9: return None
    cx /= (6*a); cy /= (6*a)
    return (cx, cy)

def transform_points(points, dx, dy, angle_deg, center):
    if not points: return []
    ang = np.deg2rad(angle_deg); c,s = np.cos(ang), np.sin(ang)
    cx, cy = center
    out = []
    for (x,y) in points:
        x0, y0 = x - cx, y - cy
        xr = x0*c - y0*s + cx + dx
        yr = x0*s + y0*c + cy + dy
        out.append((float(xr), float(yr)))
    return out

ss = st.session_state
def init():
    ss.setdefault("poly", [])
    ss.setdefault("prox_line", [])
    ss.setdefault("dist_line", [])
    ss.setdefault("cora", None)
    ss.setdefault("hinge", None)
    ss.setdefault("segment", "distal")
    ss.setdefault("dx", 0)
    ss.setdefault("dy", 0)
    ss.setdefault("rot", 0)
    ss.setdefault("canvas_px", 900)
init()

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Tool", ["Polygon","HINGE","CORA","Proximal line","Distal line"], index=0)
ss.segment = st.sidebar.radio("Move", ["distal","proximal"], index=0, horizontal=True)
ss.dx = st.sidebar.slider("ΔX", -1000, 1000, ss.dx, 1)
ss.dy = st.sidebar.slider("ΔY", -1000, 1000, ss.dy, 1)
ss.rot = st.sidebar.slider("Rotate (°)", -180, 180, ss.rot, 1)
ss.canvas_px = st.sidebar.slider("Canvas width (px)", 600, 1600, ss.canvas_px, 50)
c1, c2, c3, c4, c5 = st.sidebar.columns(5)
if c1.button("Undo"):
    if tool == "Polygon" and ss.poly: ss.poly.pop()
    elif tool == "Proximal line" and ss.prox_line: ss.prox_line.pop()
    elif tool == "Distal line" and ss.dist_line: ss.dist_line.pop()
    elif tool == "HINGE": ss.hinge = None
    elif tool == "CORA": ss.cora = None
if c2.button("Reset poly"): ss.poly.clear()
if c3.button("Reset lines"): ss.prox_line.clear(); ss.dist_line.clear()
if c4.button("Clear centers"): ss.cora=None; ss.hinge=None
if c5.button("Reset move"): ss.dx=0; ss.dy=0; ss.rot=0

if not uploaded:
    st.info("Upload an image to start."); st.stop()

img = decode_image(uploaded.getvalue())
W, H = img.size
scale = ss.canvas_px / float(W)
disp_h = int(round(H*scale))

bg = img.copy()
d = ImageDraw.Draw(bg)
if len(ss.poly) >= 2:
    d.line(ss.poly, fill=(0,255,255,255), width=2)
    if len(ss.poly) >= 3:
        d.line([*ss.poly, ss.poly[0]], fill=(0,255,255,255), width=2)
if ss.prox_line:
    d.line(ss.prox_line, fill=(66,133,244,255), width=3)
if ss.dist_line:
    d.line(ss.dist_line, fill=(221,0,221,255), width=3)
if ss.cora:
    x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)
if ss.hinge:
    x,y=ss.hinge; d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
    d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
    d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)

canvas_res = st_canvas(
    background_image=bg,
    width=ss.canvas_px,
    height=disp_h,
    drawing_mode="polyline" if tool in ["Polygon","Proximal line","Distal line"] else "transform",
    stroke_width=2,
    stroke_color="#00FFFF" if tool=="Polygon" else ("#4285F4" if tool=="Proximal line" else "#DD00DD"),
    update_streamlit=True,
    key="main_canvas",
)

def to_orig(x,y):
    return (x/scale, y/scale)

if canvas_res.json_data is not None and "objects" in canvas_res.json_data:
    objs = canvas_res.json_data["objects"]
    if objs:
        last = objs[-1]
        if tool in ["Polygon","Proximal line","Distal line"] and last.get("type") in ["line","polyline","path"]:
            raw = last.get("path", [])
            pts = []
            for cmd in raw:
                if len(cmd) >= 3:
                    _, x, y = cmd[:3]
                    ox, oy = to_orig(x, y)
                    pts.append((ox, oy))
            if tool == "Polygon":
                ss.poly = pts
            elif tool == "Proximal line":
                if len(pts) >= 2: ss.prox_line = [pts[0], pts[-1]]
            elif tool == "Distal line":
                if len(pts) >= 2: ss.dist_line = [pts[0], pts[-1]]

center = ss.hinge or ss.cora or centroid(ss.poly)
if len(ss.poly) >= 3 and center is not None:
    m = polygon_mask(img.size, ss.poly)
    inv = ImageOps.invert(m)
    prox = Image.new("RGBA", img.size, (0,0,0,0)); prox.paste(img, (0,0), inv)
    dist = Image.new("RGBA", img.size, (0,0,0,0)); dist.paste(img, (0,0), m)
    moving = dist if ss.segment == "distal" else prox
    fixed  = prox if ss.segment == "distal" else dist
    moved = apply_affine(moving, ss.dx, ss.dy, ss.rot, center)
    out = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", img.size, (0,0,0,0)), fixed), moved)
    # redraw lines with correct following
    draw2 = ImageDraw.Draw(out)
    if len(ss.dist_line) == 2:
        p = transform_points(ss.dist_line, ss.dx, ss.dy, ss.rot, center) if ss.segment=="distal" else ss.dist_line
        draw2.line(p, fill=(221,0,221,255), width=3)
    if len(ss.prox_line) == 2:
        p = transform_points(ss.prox_line, ss.dx, ss.dy, ss.rot, center) if ss.segment=="proximal" else ss.prox_line
        draw2.line(p, fill=(66,133,244,255), width=3)
    st.image(out.resize((ss.canvas_px, disp_h), Image.NEAREST), use_container_width=True)
else:
    st.info("Draw polygon (≥3) and set HINGE/CORA; live drawing is active.")
