
import io
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

st.set_page_config(page_title="Bone Ninja — Fast & Accurate", layout="wide")

# Colors (RGBA)
CYAN = (0, 255, 255, 255)
RED = (255, 0, 0, 255)
GREEN = (0, 200, 0, 255)
BLUE = (66, 133, 244, 255)
MAGENTA = (221, 0, 221, 255)
ORANGE = (255, 165, 0, 255)
YELLOW = (255, 255, 0, 255)

@dataclass
class TransformParams:
    mode: str  # "distal" or "proximal"
    dx: float
    dy: float
    rotate_deg: float
    center: Tuple[float,float]

def decode_image(file_bytes) -> Image.Image:
    # Fix EXIF orientation so displayed image and coordinate space are aligned
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

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

def rotate_point(p, center, angle_deg):
    ang = np.deg2rad(angle_deg)
    c, s = np.cos(ang), np.sin(ang)
    x, y = p; cx, cy = center
    x0, y0 = x - cx, y - cy
    xr = x0*c - y0*s + cx
    yr = x0*s + y0*c + cy
    return (float(xr), float(yr))

def transform_points(points, dx, dy, angle_deg, center):
    out = []
    for (px,py) in points:
        xr, yr = rotate_point((px,py), center, angle_deg)
        out.append((xr + dx, yr + dy))
    return out

# ---------------- State ----------------
ss = st.session_state
def init_state():
    ss.setdefault("poly_points", [])
    ss.setdefault("cora_pt", None)
    ss.setdefault("hinge_pt", None)
    ss.setdefault("ruler_points", [])
    ss.setdefault("prox_line", [])   # up to 2 points
    ss.setdefault("dist_line", [])   # up to 2 points
    ss.setdefault("display_width", 1100)
    ss.setdefault("last_pt", None)
init_state()

# ---------------- Sidebar ----------------
st.sidebar.title("Tools")
uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Active tool", ["Polygon","CORA","HINGE","Proximal line","Distal line","Ruler"], index=0)

segment_choice = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True, index=0)
center_mode = st.sidebar.radio("Rotation center", ["HINGE","CORA","Polygon centroid"], index=0)

ss.display_width = st.sidebar.slider("Preview width (px)", 600, 1800, ss.display_width, 50)

dx = st.sidebar.slider("ΔX (px)", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY (px)", -1000, 1000, 0, 1)
rotate_deg = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)

c1, c2, c3, c4, c5 = st.sidebar.columns(5)
with c1:
    if st.button("Undo"):
        if tool == "Polygon" and ss.poly_points: ss.poly_points.pop()
        elif tool == "Proximal line" and ss.prox_line: ss.prox_line.pop()
        elif tool == "Distal line" and ss.dist_line: ss.dist_line.pop()
        elif tool == "Ruler" and ss.ruler_points: ss.ruler_points.pop()
        elif tool == "CORA": ss.cora_pt = None
        elif tool == "HINGE": ss.hinge_pt = None
with c2:
    if st.button("Reset polygon"): ss.poly_points.clear()
with c3:
    if st.button("Reset lines"): ss.prox_line.clear(); ss.dist_line.clear()
with c4:
    if st.button("Clear centers"): ss.cora_pt=None; ss.hinge_pt=None
with c5:
    if st.button("Clear ruler"): ss.ruler_points.clear()

if uploaded is None:
    st.info("Upload an image to begin."); st.stop()

@st.cache_data(show_spinner=False)
def load_img(data: bytes):
    img = decode_image(data)
    return img, img.size

base_img, (W, H) = load_img(uploaded.getvalue())

# --- Accurate scaling (independent x/y); no further resizing by Streamlit ---
disp_w = min(ss.display_width, W)  # don't upscale beyond native to keep accuracy
scale_x = disp_w / float(W)
disp_h = int(round(H * scale_x))
scale_y = disp_h / float(H)

preview = base_img.copy()
d = ImageDraw.Draw(preview)

# Draw overlays (persistent across tools)
if ss.poly_points:
    d.line(ss.poly_points, fill=CYAN, width=2, joint="curve")
    if len(ss.poly_points) >= 3:
        d.line([*ss.poly_points, ss.poly_points[0]], fill=CYAN, width=2)
# lines
if ss.prox_line:
    if len(ss.prox_line) == 1:
        p = ss.prox_line[0]; d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=BLUE)
    elif len(ss.prox_line) == 2:
        d.line(ss.prox_line, fill=BLUE, width=3)
        for p in ss.prox_line: d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=BLUE)
if ss.dist_line:
    if len(ss.dist_line) == 1:
        p = ss.dist_line[0]; d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=MAGENTA)
    elif len(ss.dist_line) == 2:
        d.line(ss.dist_line, fill=MAGENTA, width=3)
        for p in ss.dist_line: d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=MAGENTA)
# ruler
if ss.ruler_points:
    if len(ss.ruler_points) == 1:
        r = ss.ruler_points[0]; d.ellipse([r[0]-4,r[1]-4,r[0]+4,r[1]+4], fill=RED)
    elif len(ss.ruler_points) == 2:
        d.line(ss.ruler_points, fill=RED, width=2)
        for r in ss.ruler_points: d.ellipse([r[0]-4,r[1]-4,r[0]+4,r[1]+4], fill=RED)
# centers
if ss.cora_pt:
    x,y = ss.cora_pt; d.ellipse([x-6,y-6,x+6,y+6], outline=GREEN, width=2)
if ss.hinge_pt:
    x,y = ss.hinge_pt
    d.ellipse([x-7,y-7,x+7,y+7], outline=ORANGE, width=3)
    d.line([(x-12,y),(x+12,y)], fill=ORANGE, width=1)
    d.line([(x,y-12),(x,y+12)], fill=ORANGE, width=1)

# High-speed resize (NEAREST) to avoid anti-aliased offset illusion
disp_img = preview.resize((disp_w, disp_h), Image.NEAREST)

st.caption("Crosshair cursor: click precisely where you want (no modebar; persistent tools).")
res = streamlit_image_coordinates(
    disp_img, key="imgclick_stable", width=disp_w, show_coordinates=False, css={"cursor":"crosshair"}
)

def to_orig(pt_disp):
    return (float(pt_disp[0])/scale_x, float(pt_disp[1])/scale_y)

if res is not None and "x" in res and "y" in res:
    pt = to_orig((res["x"], res["y"]))
    ss.last_pt = pt
    if tool == "Polygon":
        ss.poly_points.append(pt)
    elif tool == "CORA":
        ss.cora_pt = pt
    elif tool == "HINGE":
        ss.hinge_pt = pt
    elif tool == "Proximal line":
        if len(ss.prox_line) >= 2: ss.prox_line.clear()
        ss.prox_line.append(pt)
    elif tool == "Distal line":
        if len(ss.dist_line) >= 2: ss.dist_line.clear()
        ss.dist_line.append(pt)
    elif tool == "Ruler":
        if len(ss.ruler_points) >= 2: ss.ruler_points.clear()
        ss.ruler_points.append(pt)
    st.success(f"Added {tool} point at ({pt[0]:.1f}, {pt[1]:.1f})")

# Measurements
st.subheader("Measurements")
msgs = []
if len(ss.ruler_points) == 2: msgs.append(f"Ruler: {length_of_line(*ss.ruler_points):.2f} px")
if len(ss.prox_line) == 2: msgs.append(f"Proximal line: {length_of_line(*ss.prox_line):.2f} px")
if len(ss.dist_line) == 2: msgs.append(f"Distal line: {length_of_line(*ss.dist_line):.2f} px")
for m in msgs: st.info(m)
if not msgs: st.caption("Use Ruler or Proximal/Distal lines to see lengths.")

# Determine rotation center
center = None
if center_mode == "HINGE" and ss.hinge_pt: center = ss.hinge_pt
elif center_mode == "CORA" and ss.cora_pt: center = ss.cora_pt
else: center = centroid_of_polygon(ss.poly_points) or ss.cora_pt or ss.hinge_pt

st.header("Preview and Export")
if len(ss.poly_points) >= 3 and center is not None:
    mask_poly = polygon_mask(base_img.size, ss.poly_points)
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

    # Redraw lines
    draw2 = ImageDraw.Draw(out_img)
    # Distal line follows distal fragment if it moved
    if len(ss.dist_line) == 2:
        if segment_choice == "distal":
            p_trans = transform_points(ss.dist_line, dx, dy, rotate_deg, center)
            draw2.line(p_trans, fill=MAGENTA, width=3)
            for p in p_trans: draw2.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=MAGENTA)
        else:
            draw2.line(ss.dist_line, fill=MAGENTA, width=3)
            for p in ss.dist_line: draw2.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=MAGENTA)

    if len(ss.prox_line) == 2:
        if segment_choice == "proximal":
            p_trans = transform_points(ss.prox_line, dx, dy, rotate_deg, center)
            draw2.line(p_trans, fill=BLUE, width=3)
            for p in p_trans: draw2.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=BLUE)
        else:
            draw2.line(ss.prox_line, fill=BLUE, width=3)
            for p in ss.prox_line: draw2.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=BLUE)

    # Center marker
    x,y = center; draw2.ellipse([x-6,y-6,x+6,y+6], outline=YELLOW, width=2)

    st.image(out_img.resize((disp_w, disp_h), Image.NEAREST),
             caption=f"Transformed around {center} — {segment_choice} Δ=({dx},{dy}) θ={rotate_deg}°",
             use_container_width=True)

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
    st.info("Draw a polygon (≥3 points) and pick a rotation center (HINGE/CORA/centroid) to preview transform.")
