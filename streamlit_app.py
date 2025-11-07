
import io
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

st.set_page_config(page_title="Bone Ninja — Streamlit (click tools + hinge)", layout="wide")

# Colors (RGBA)
CYAN = (0, 255, 255, 255)
RED = (255, 0, 0, 255)
GREEN = (0, 200, 0, 255)
BLUE = (66, 133, 244, 255)
MAGENTA = (221, 0, 221, 255)
ORANGE = (255, 165, 0, 255)
GREY = (180,180,180,255)

@dataclass
class TransformParams:
    mode: str  # "distal" or "proximal"
    dx: float
    dy: float
    rotate_deg: float
    center: Tuple[float,float]

def pil_from_bytes(file_bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGBA")

def polygon_mask(size: Tuple[int,int], points: List[Tuple[float,float]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=255, outline=255)
    return mask

def apply_affine(img: Image.Image, dx: float, dy: float, rot_deg: float, center: Tuple[float,float]) -> Image.Image:
    rotated = img.rotate(rot_deg, resample=Image.BICUBIC, center=center, expand=False)
    canvas = Image.new("RGBA", img.size, (0,0,0,0))
    canvas.alpha_composite(rotated, (int(round(dx)), int(round(dy))))
    return canvas

def paste_with_mask(base: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
    out = base.copy()
    out.paste(overlay, (0,0), mask)
    return out

def centroid_of_polygon(pts: List[Tuple[float,float]]) -> Optional[Tuple[float,float]]:
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

def length_of_line(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))

# ---------------- State ----------------
ss = st.session_state
def init_state():
    ss.setdefault("poly_points", [])
    ss.setdefault("cora_pt", None)
    ss.setdefault("hinge_pt", None)
    ss.setdefault("ruler_points", [])
    ss.setdefault("prox_line", [])   # up to 2 points
    ss.setdefault("dist_line", [])   # up to 2 points
    ss.setdefault("last_click", None)
init_state()

# ---------------- Sidebar ----------------
st.sidebar.title("Workflow")
st.sidebar.markdown("""
Click to add points. This build supports:
- **Polygon** (osteotomy)
- **CORA** center
- **HINGE** (rotation center — **recommended**)
- **Proximal line** and **Distal line**
- **Ruler** (2‑point)
""")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Active tool", ["Polygon","CORA","HINGE","Proximal line","Distal line","Ruler"], horizontal=False, index=0)
segment_choice = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True, index=0)

st.sidebar.subheader("Rotation center")
center_mode = st.sidebar.radio("Use center from", ["HINGE","CORA","Polygon centroid"], index=0, horizontal=False)

st.sidebar.subheader("Translation")
dx = st.sidebar.slider("ΔX (px)", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY (px)", -1000, 1000, 0, 1)
st.sidebar.subheader("Rotation")
rotate_deg = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)

b1, b2, b3, b4, b5 = st.sidebar.columns(5)
with b1:
    if st.button("Undo"):
        if tool == "Polygon" and ss.poly_points: ss.poly_points.pop()
        elif tool == "Ruler" and ss.ruler_points: ss.ruler_points.pop()
        elif tool == "Proximal line" and ss.prox_line: ss.prox_line.pop()
        elif tool == "Distal line" and ss.dist_line: ss.dist_line.pop()
        elif tool == "CORA": ss.cora_pt=None
        elif tool == "HINGE": ss.hinge_pt=None
with b2:
    if st.button("Reset polygon"): ss.poly_points.clear()
with b3:
    if st.button("Reset lines"): ss.prox_line.clear(); ss.dist_line.clear()
with b4:
    if st.button("Clear centers"): ss.cora_pt=None; ss.hinge_pt=None
with b5:
    if st.button("Clear ruler"): ss.ruler_points.clear()

if uploaded is None:
    st.info("Upload an image to begin."); st.stop()

base_img = pil_from_bytes(uploaded.getvalue())
W, H = base_img.size

# --------- Draw overlays for preview (also the clickable image) ---------
preview = base_img.copy()
draw = ImageDraw.Draw(preview)

# polygon
if len(ss.poly_points) >= 1:
    draw.line(ss.poly_points, fill=CYAN, width=2, joint="curve")
    if len(ss.poly_points) >= 3:
        draw.line([*ss.poly_points, ss.poly_points[0]], fill=CYAN, width=2)

# lines (proximal in BLUE, distal in MAGENTA)
if len(ss.prox_line) == 1:
    p = ss.prox_line[0]
    draw.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3], fill=BLUE)
elif len(ss.prox_line) == 2:
    draw.line(ss.prox_line, fill=BLUE, width=3)
    for p in ss.prox_line:
        draw.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3], fill=BLUE)
if len(ss.dist_line) == 1:
    p = ss.dist_line[0]
    draw.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3], fill=MAGENTA)
elif len(ss.dist_line) == 2:
    draw.line(ss.dist_line, fill=MAGENTA, width=3)
    for p in ss.dist_line:
        draw.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3], fill=MAGENTA)

# ruler
if len(ss.ruler_points) == 1:
    r1 = ss.ruler_points[0]
    draw.ellipse([r1[0]-3, r1[1]-3, r1[0]+3, r1[1]+3], fill=RED)
elif len(ss.ruler_points) == 2:
    draw.line(ss.ruler_points, fill=RED, width=2)
    for r in ss.ruler_points:
        draw.ellipse([r[0]-3, r[1]-3, r[0]+3, r[1]+3], fill=RED)

# centers
if ss.cora_pt:
    x,y = ss.cora_pt
    draw.ellipse([x-6,y-6,x+6,y+6], outline=GREEN, width=2)
if ss.hinge_pt:
    x,y = ss.hinge_pt
    draw.ellipse([x-7,y-7,x+7,y+7], outline=ORANGE, width=3)
    draw.line([(x-12,y),(x+12,y)], fill=ORANGE, width=1)
    draw.line([(x,y-12),(x,y+12)], fill=ORANGE, width=1)

st.caption("Click on the image below to add to the active tool.")
res = streamlit_image_coordinates(preview, key="imgclick_v2", width=min(1100, W))

if res is not None and "x" in res and "y" in res:
    pt = (float(res["x"]), float(res["y"]))
    ss.last_click = pt
    if tool == "Polygon":
        ss.poly_points.append(pt)
    elif tool == "CORA":
        ss.cora_pt = pt
    elif tool == "HINGE":
        ss.hinge_pt = pt
    elif tool == "Ruler":
        if len(ss.ruler_points) >= 2: ss.ruler_points.clear()
        ss.ruler_points.append(pt)
    elif tool == "Proximal line":
        if len(ss.prox_line) >= 2: ss.prox_line.clear()
        ss.prox_line.append(pt)
    elif tool == "Distal line":
        if len(ss.dist_line) >= 2: ss.dist_line.clear()
        ss.dist_line.append(pt)
    st.success(f"Added point x={pt[0]:.1f}, y={pt[1]:.1f} to {tool}")

# Measurements
st.subheader("Measurements")
meas_msgs = []
if len(ss.ruler_points) == 2:
    meas_msgs.append(f"Ruler: {length_of_line(*ss.ruler_points):.2f} px")
if len(ss.prox_line) == 2:
    meas_msgs.append(f"Proximal line length: {length_of_line(*ss.prox_line):.2f} px")
if len(ss.dist_line) == 2:
    meas_msgs.append(f"Distal line length: {length_of_line(*ss.dist_line):.2f} px")
if meas_msgs:
    for m in meas_msgs: st.info(m)
else:
    st.caption("Use Ruler or draw Proximal/Distal lines to see measurements.")

# Determine rotation center
center = None
if center_mode == "HINGE" and ss.hinge_pt:
    center = ss.hinge_pt
elif center_mode == "CORA" and ss.cora_pt:
    center = ss.cora_pt
elif center_mode == "Polygon centroid":
    center = centroid_of_polygon(ss.poly_points)
if center is None and ss.cora_pt:
    center = ss.cora_pt
if center is None:
    c = centroid_of_polygon(ss.poly_points)
    if c: center = c

# Transform preview + export
st.header("Preview and Export")
if len(ss.poly_points) >= 3 and center is not None:
    poly_pts = ss.poly_points
    mask_poly = polygon_mask(base_img.size, poly_pts)
    mask_inv = ImageOps.invert(mask_poly)

    proximal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
    distal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
    proximal_piece = paste_with_mask(proximal_piece, base_img, mask_inv)
    distal_piece = paste_with_mask(distal_piece, base_img, mask_poly)

    moving = distal_piece if segment_choice == "distal" else proximal_piece
    fixed = proximal_piece if segment_choice == "distal" else distal_piece

    moved = apply_affine(moving, dx=dx, dy=dy, rot_deg=rotate_deg, center=center)

    composed = Image.new("RGBA", base_img.size, (0,0,0,0))
    out_img = Image.alpha_composite(Image.alpha_composite(composed, fixed), moved)

    # draw overlays on top for preview (center marker)
    preview2 = out_img.copy()
    d2 = ImageDraw.Draw(preview2)
    if center:
        x,y = center
        d2.ellipse([x-6,y-6,x+6,y+6], outline=(255,255,0,255), width=2)
    st.image(preview2, caption=f"Transformed around {center_mode} ({segment_choice} moved by Δ=({dx},{dy}), θ={rotate_deg}°)", use_container_width=True)

    # Export
    params = dict(
        mode=segment_choice, dx=dx, dy=dy, rotate_deg=rotate_deg,
        rotation_center=center,
        polygon_points=ss.poly_points,
        cora=ss.cora_pt, hinge=ss.hinge_pt,
        proximal_line=ss.prox_line, distal_line=ss.dist_line
    )
    df = pd.DataFrame([params])
    st.download_button("Download parameters CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="osteotomy_params.csv", mime="text/csv")
    buf = io.BytesIO(); out_img.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)", data=buf.getvalue(),
                       file_name="osteotomy_transformed.png", mime="image/png")
else:
    st.info("Need an osteotomy polygon (≥3 points) and a rotation center (HINGE/CORA/centroid).")
