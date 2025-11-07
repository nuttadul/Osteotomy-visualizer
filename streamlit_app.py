
import io
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd

st.set_page_config(page_title="Bone Ninja — Plotly Edition", layout="wide")

CYAN = "cyan"

@dataclass
class TransformParams:
    mode: str  # "proximal" or "distal"
    dx: float
    dy: float
    rotate_deg: float

@dataclass
class MeasurementRecord:
    timestamp: float
    tool: str  # "ruler" or "angle"
    values: dict

def pil_from_bytes(file_bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
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

def angle_from_three_points(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def length_of_line(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))

# ---------------- State ----------------
if "poly_points" not in st.session_state:
    st.session_state.poly_points: List[Tuple[float,float]] = []
if "ruler_points" not in st.session_state:
    st.session_state.ruler_points: List[Tuple[float,float]] = []
if "angle_points" not in st.session_state:
    st.session_state.angle_points: List[Tuple[float,float]] = []
if "measurements" not in st.session_state:
    st.session_state.measurements: List[MeasurementRecord] = []
if "px_per_mm" not in st.session_state:
    st.session_state.px_per_mm = None
if "params" not in st.session_state:
    st.session_state.params = TransformParams(mode="distal", dx=0.0, dy=0.0, rotate_deg=0.0)

# ---------------- Sidebar ----------------
st.sidebar.title("Workflow")
st.sidebar.markdown("1. Upload image\n2. Choose tool\n3. Click on image to add points\n4. Close polygon\n5. Adjust transform\n6. Export")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Tool", ["Osteotomy polygon","Ruler (2 clicks)","Angle (3 clicks)"])
segment_choice = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True)

st.sidebar.subheader("Translation")
dx = st.sidebar.slider("ΔX (pixels)", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY (pixels)", -1000, 1000, 0, 1)
st.sidebar.subheader("Rotation")
rotate_deg = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)
if st.sidebar.button("Reset transforms"):
    st.session_state.params = TransformParams(segment_choice, 0.0, 0.0, 0.0)
else:
    st.session_state.params = TransformParams(segment_choice, float(dx), float(dy), float(rotate_deg))

st.sidebar.subheader("Calibration (optional)")
cal_mm = st.sidebar.number_input("Known distance (mm) for last ruler", min_value=0.0, value=0.0, step=0.1)
apply_cal = st.sidebar.button("Apply calibration")

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

base_img = pil_from_bytes(uploaded.getvalue())
W, H = base_img.size

# ---------------- Figure ----------------
# Plotly axes in pixel space; y reversed to match image coordinates
fig = go.Figure()
fig.update_xaxes(range=[0, W], constrain="domain", visible=False)
fig.update_yaxes(range=[H, 0], scaleanchor="x", scaleratio=1, visible=False)
fig.add_layout_image(dict(source=base_img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", layer="below"))

# Add overlays
pp = st.session_state.poly_points
if len(pp) >= 1:
    xs, ys = zip(*pp)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=CYAN, width=2), marker=dict(size=6, color=CYAN), name="polygon"))
if len(pp) >= 3:
    # Close preview
    xs, ys = zip(*([*pp, pp[0]]))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=CYAN, width=2), showlegend=False))

rp = st.session_state.ruler_points
if len(rp) >= 1:
    xs, ys = zip(*rp)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(width=2), name="ruler"))

ap = st.session_state.angle_points
if len(ap) >= 1:
    xs, ys = zip(*ap[:3])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(width=2), name="angle"))

fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), dragmode="pan")

st.caption("Click on the image to add points. Use the buttons below to manage points.")
selected = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="plot")

colA, colB, colC, colD = st.columns(4)
with colA:
    if st.button("Add point from last click"):
        if selected:
            x = float(selected[-1]["x"]); y = float(selected[-1]["y"])
            if tool.startswith("Osteotomy"):
                st.session_state.poly_points.append((x,y))
            elif tool.startswith("Ruler"):
                pts = st.session_state.ruler_points
                if len(pts) >= 2: pts.clear()
                pts.append((x,y))
            else:
                pts = st.session_state.angle_points
                if len(pts) >= 3: pts.clear()
                pts.append((x,y))
with colB:
    if st.button("Undo last point"):
        if tool.startswith("Osteotomy") and st.session_state.poly_points:
            st.session_state.poly_points.pop()
        elif tool.startswith("Ruler") and st.session_state.ruler_points:
            st.session_state.ruler_points.pop()
        elif tool.startswith("Angle") and st.session_state.angle_points:
            st.session_state.angle_points.pop()
with colC:
    poly_closed = st.button("Close polygon")
with colD:
    if st.button("Clear all points"):
        st.session_state.poly_points.clear()
        st.session_state.ruler_points.clear()
        st.session_state.angle_points.clear()

# Measurements
st.subheader("Measurements")
if apply_cal and cal_mm > 0 and len(st.session_state.ruler_points) == 2:
    px = length_of_line(*st.session_state.ruler_points)
    st.session_state.px_per_mm = px / cal_mm
    st.success(f"Calibration set: {st.session_state.px_per_mm:.3f} px/mm")

meas_text = ""
if len(st.session_state.ruler_points) == 2:
    px = length_of_line(*st.session_state.ruler_points)
    meas_text += f"Ruler: {px:.2f} px"
    if st.session_state.px_per_mm:
        mm = px / st.session_state.px_per_mm
        meas_text += f"  ({mm:.2f} mm)"
if len(st.session_state.angle_points) == 3:
    ang = angle_from_three_points(st.session_state.angle_points[0], st.session_state.angle_points[1], st.session_state.angle_points[2])
    meas_text += (" | " if meas_text else "") + f"Angle: {ang:.2f}°"
if meas_text:
    st.info(meas_text)
else:
    st.caption("Use the Ruler and Angle tools to add points.")

# Transform preview + export
st.header("Preview and Export")
params = asdict(st.session_state.params)
params["polygon_points"] = st.session_state.poly_points

if len(st.session_state.poly_points) >= 3 and (poly_closed or True):
    poly_pts = [(float(x), float(y)) for x,y in st.session_state.poly_points]

    mask_poly = polygon_mask(base_img.size, poly_pts)
    mask_inv = ImageOps.invert(mask_poly)

    proximal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
    distal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
    proximal_piece = paste_with_mask(proximal_piece, base_img, mask_inv)
    distal_piece = paste_with_mask(distal_piece, base_img, mask_poly)

    seg_mask = mask_poly if st.session_state.params.mode == "distal" else mask_inv

    arr = np.array(seg_mask) / 255.0
    ys, xs = np.nonzero(arr > 0.5)
    if len(xs) == 0:
        center = (W//2, H//2)
    else:
        center = (float(xs.mean()), float(ys.mean()))

    moving = distal_piece if st.session_state.params.mode == "distal" else proximal_piece
    fixed = proximal_piece if st.session_state.params.mode == "distal" else distal_piece

    moved = apply_affine(moving, dx=st.session_state.params.dx, dy=st.session_state.params.dy, rot_deg=st.session_state.params.rotate_deg, center=center)

    composed = Image.new("RGBA", base_img.size, (0,0,0,0))
    composed = Image.alpha_composite(composed, fixed)
    composed = Image.alpha_composite(composed, moved)

    st.image(composed, caption=f"Transformed ({st.session_state.params.mode} moved)", use_container_width=True)

    if st.session_state.px_per_mm:
        params["px_per_mm"] = st.session_state.px_per_mm
    df_params = pd.DataFrame([params]).astype(object)
    st.download_button("Download parameters CSV", data=df_params.to_csv(index=False).encode("utf-8"), file_name="osteotomy_params.csv", mime="text/csv")

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)", data=buf.getvalue(), file_name="osteotomy_transformed.png", mime="image/png")
else:
    st.caption("Add at least 3 polygon points and click Close polygon to preview transform.")
