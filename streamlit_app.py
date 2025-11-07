
import io
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd

st.set_page_config(page_title="Bone Ninja — Streamlit (Plotly)", layout="wide")

CYAN = "cyan"

@dataclass
class TransformParams:
    mode: str  # "proximal" or "distal"
    dx: float
    dy: float
    rotate_deg: float

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
if "px_per_mm" not in st.session_state:
    st.session_state.px_per_mm = None

# ---------------- Sidebar ----------------
st.sidebar.title("Workflow")
st.sidebar.markdown("1. Upload image\n2. Use toolbar draw tools\n3. Click **Use last closed path** to set osteotomy polygon\n4. Choose CORA (click on image)\n5. Adjust ΔX / ΔY / Rotate\n6. Export")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
segment_choice = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True)

st.sidebar.subheader("Translation")
dx = st.sidebar.slider("ΔX (px)", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY (px)", -1000, 1000, 0, 1)
st.sidebar.subheader("Rotation")
rotate_deg = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

base_img = pil_from_bytes(uploaded.getvalue())
W, H = base_img.size

# ---------------- Figure ----------------
fig = go.Figure()
fig.update_xaxes(range=[0, W], constrain="domain", visible=False)
fig.update_yaxes(range=[H, 0], scaleanchor="x", scaleratio=1, visible=False)
fig.add_layout_image(dict(source=base_img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", layer="below"))
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), dragmode="pan")

# Modebar drawing tools
config = {
    "modeBarButtonsToAdd": ["drawclosedpath", "drawcircle", "drawline", "eraseshape"],
    "displaylogo": False
}

st.caption("Toolbar: draw **closed path** for osteotomy, **circle** for CORA, **line** for ruler/markers. Then use the buttons below.")
events = plotly_events(fig, events=["relayout", "click"], click_event=True, select_event=False, override_height=None, override_width="100%", config=config, key="plot_ev")

col1, col2, col3, col4 = st.columns(4)
with col1:
    use_last_poly = st.button("Use last closed path (osteotomy)")
with col2:
    set_cora_from_click = st.button("Set CORA = last click")
with col3:
    set_ruler_from_last_line = st.button("Use last drawn line as ruler")
with col4:
    clear_all = st.button("Clear overlays")

# Parse relayout to get latest shape path
last_path = None
last_circle = None
last_line = None
if events:
    rel = events[-1].get("relayout", {})
    for k, v in rel.items():
        if isinstance(v, dict) and "path" in v:  # shape dict
            if "Z" in v["path"]:
                last_path = v["path"]
        elif isinstance(v, str) and ("M" in v or "L" in v) and "path" in k:
            if "Z" in v:
                last_path = v
        # circle is given by 'x0','x1','y0','y1' in shape; we ignore for simplicity

# Click to set CORA
cora_pt = None
if set_cora_from_click and events and "x" in events[-1] and "y" in events[-1]:
    cora_pt = (float(events[-1]["x"]), float(events[-1]["y"]))
elif "cora_pt" in st.session_state:
    cora_pt = st.session_state["cora_pt"]

def path_to_points(path: str) -> List[Tuple[float,float]]:
    pairs = re.findall(r'(-?\d+\.?\d*),(-?\d+\.?\d*)', path)
    return [(float(x), float(y)) for x,y in pairs]

if use_last_poly and last_path:
    st.session_state.poly_points = path_to_points(last_path)

if set_ruler_from_last_line and last_path:
    pts = path_to_points(last_path)
    if len(pts) >= 2:
        st.session_state.ruler_points = [pts[0], pts[-1]]

if clear_all:
    st.session_state.poly_points.clear()
    st.session_state.ruler_points.clear()
    st.session_state.angle_points.clear()
    st.session_state.pop("cora_pt", None)
    cora_pt = None

# Show overlays
ov = go.Figure()
ov.update_xaxes(range=[0, W], visible=False)
ov.update_yaxes(range=[H, 0], scaleanchor="x", scaleratio=1, visible=False)
ov.add_layout_image(dict(source=base_img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", layer="below"))
if st.session_state.poly_points:
    xs, ys = zip(*st.session_state.poly_points)
    ov.add_trace(go.Scatter(x=list(xs)+[xs[0]], y=list(ys)+[ys[0]], mode="lines+markers",
                            line=dict(color=CYAN, width=2), marker=dict(size=6, color=CYAN), name="polygon"))
if st.session_state.ruler_points:
    xs, ys = zip(*st.session_state.ruler_points)
    ov.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="ruler"))
if cora_pt:
    ov.add_trace(go.Scatter(x=[cora_pt[0]], y=[cora_pt[1]], mode="markers", marker=dict(size=10, symbol="circle"), name="CORA"))
st.plotly_chart(ov, use_container_width=True, config={"displaylogo": False})

# Preview / export
st.header("Preview and Export")
if st.session_state.poly_points and cora_pt:
    poly_pts = st.session_state.poly_points
    mask_poly = polygon_mask(base_img.size, poly_pts)
    mask_inv = ImageOps.invert(mask_poly)

    proximal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
    distal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
    proximal_piece = paste_with_mask(proximal_piece, base_img, mask_inv)
    distal_piece = paste_with_mask(distal_piece, base_img, mask_poly)

    seg_mask = mask_poly if segment_choice == "distal" else mask_inv
    arr = np.array(seg_mask) / 255.0
    ys, xs = np.nonzero(arr > 0.5)
    center = (float(xs.mean()), float(ys.mean())) if len(xs) else (W/2, H/2)

    moving = distal_piece if segment_choice == "distal" else proximal_piece
    fixed = proximal_piece if segment_choice == "distal" else distal_piece

    moved = apply_affine(moving, dx=dx, dy=dy, rot_deg=rotate_deg, center=center)

    composed = Image.new("RGBA", base_img.size, (0,0,0,0))
    composed = Image.alpha_composite(composed, fixed)
    composed = Image.alpha_composite(composed, moved)

    st.image(composed, caption=f"Transformed ({segment_choice} moved)", use_container_width=True)

    params = dict(mode=segment_choice, dx=dx, dy=dy, rotate_deg=rotate_deg, polygon_points=poly_pts, cora=cora_pt)
    df_params = pd.DataFrame([params])
    st.download_button("Download parameters CSV", data=df_params.to_csv(index=False).encode("utf-8"), file_name="osteotomy_params.csv", mime="text/csv")

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)", data=buf.getvalue(), file_name="osteotomy_transformed.png", mime="image/png")
else:
    st.info("Draw polygon and set CORA (click) to preview transform.")
