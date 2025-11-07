
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

st.set_page_config(page_title="Bone Ninja — Freehand (Plotly draw tools)", layout="wide")

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

# Parse Plotly SVG path from draw tools: "M x,y L x,y L x,y Z"
def path_to_points(path: str) -> List[Tuple[float,float]]:
    # Extract all number pairs
    # Path format can include 'M','L','Q','Z'. We'll take all "number,number" pairs.
    nums = re.findall(r'(-?\d+\.?\d*),(-?\d+\.?\d*)', path)
    return [(float(x), float(y)) for x,y in nums]

# ---------------- State ----------------
if "poly_points" not in st.session_state:
    st.session_state.poly_points: List[Tuple[float,float]] = []
if "ruler_points" not in st.session_state:
    st.session_state.ruler_points: List[Tuple[float,float]] = []
if "angle_points" not in st.session_state:
    st.session_state.angle_points: List[Tuple[float,float]] = []
if "px_per_mm" not in st.session_state:
    st.session_state.px_per_mm = None
if "measure_log" not in st.session_state:
    st.session_state.measure_log = []  # simple list of dicts
if "params" not in st.session_state:
    st.session_state.params = TransformParams(mode="distal", dx=0.0, dy=0.0, rotate_deg=0.0)

# ---------------- Sidebar ----------------
st.sidebar.title("Workflow")
st.sidebar.markdown("1. Upload image\n2. Use the modebar to **Draw closed path** (lasso)\n3. Click **Use last closed path**\n4. Choose Ruler/Angle tools\n5. Adjust ΔX ΔY Rotate\n6. Export")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
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
fig = go.Figure()
fig.update_xaxes(range=[0, W], constrain="domain", visible=False)
fig.update_yaxes(range=[H, 0], scaleanchor="x", scaleratio=1, visible=False)
fig.add_layout_image(dict(source=base_img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", layer="below"))
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), dragmode="pan")

# Modebar drawing tools (Plotly will let users draw paths/lines)
config = {
    "modeBarButtonsToAdd": ["drawclosedpath", "drawline", "eraseshape"],
    "displaylogo": False
}

st.caption("Use the toolbar on the top-right of the chart: select **Draw closed path**, sketch your lasso, then click **Use last closed path**.")
events = plotly_events(fig, events=["relayout"], click_event=False, select_event=False, override_height=None, override_width="100%", config=config, key="plot")

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
with col_btn1:
    use_last = st.button("Use last closed path")
with col_btn2:
    clear_poly = st.button("Clear polygon")
with col_btn3:
    add_ruler = st.button("Add ruler from last drawn line")
with col_btn4:
    clear_measure = st.button("Clear ruler/angle")

# Inspect relayout events to find shapes (last one wins)
last_path = None
if events:
    rel = events[-1].get("relayout", {})
    # Path can appear under 'shapes[N].path' or 'shapes' changes
    # Scan all keys for a path value
    for k, v in rel.items():
        if isinstance(v, dict) and "path" in v:
            last_path = v["path"]
        elif isinstance(v, str) and ("M" in v or "L" in v) and "path" in k:
            last_path = v

# Buttons actions
if use_last and last_path:
    pts = path_to_points(last_path)
    if len(pts) >= 3:
        st.session_state.poly_points = pts

if clear_poly:
    st.session_state.poly_points.clear()

if clear_measure:
    st.session_state.ruler_points.clear()
    st.session_state.angle_points.clear()
    st.session_state.measure_log.clear()

if add_ruler and last_path:
    # Take first and last point from path as a quick ruler
    pts = path_to_points(last_path)
    if len(pts) >= 2:
        st.session_state.ruler_points = [pts[0], pts[-1]]

# Measurements UI (clickless simple text boxes for angle, or reuse ruler line above)
st.subheader("Measurements")
c1, c2 = st.columns(2)
with c1:
    if st.session_state.ruler_points:
        px = length_of_line(*st.session_state.ruler_points)
        msg = f"Ruler: {px:.2f} px"
        if apply_cal and cal_mm > 0:
            st.session_state.px_per_mm = px / cal_mm
        if st.session_state.px_per_mm:
            msg += f"  ({px / st.session_state.px_per_mm:.2f} mm)"
        st.info(msg)
    else:
        st.caption("Use 'Add ruler from last drawn line' to set a ruler.")
with c2:
    # Angle: allow manual three points input quickly
    ax = st.text_input("Angle points (x1,y1 ; x2,y2 ; x3,y3)", value="")
    if ax:
        try:
            ptxt = [p.strip() for p in ax.split(";")]
            def parse_pair(s):
                a,b = s.split(",")
                return (float(a), float(b))
            ap = [parse_pair(ptxt[0]), parse_pair(ptxt[1]), parse_pair(ptxt[2])]
            st.session_state.angle_points = ap
            ang = angle_from_three_points(*ap)
            st.info(f"Angle: {ang:.2f}°")
        except Exception:
            st.warning("Could not parse angle points. Format: x1,y1 ; x2,y2 ; x3,y3")

# Draw overlays preview
overlay_fig = go.Figure()
overlay_fig.update_xaxes(range=[0, W], visible=False)
overlay_fig.update_yaxes(range=[H, 0], scaleanchor="x", scaleratio=1, visible=False)
overlay_fig.add_layout_image(dict(source=base_img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", layer="below"))
if len(st.session_state.poly_points) >= 2:
    xs, ys = zip(*st.session_state.poly_points)
    overlay_fig.add_trace(go.Scatter(x=list(xs)+[xs[0]] if len(xs)>=3 else xs,
                                     y=list(ys)+[ys[0]] if len(ys)>=3 else ys,
                                     mode="lines+markers",
                                     line=dict(color=CYAN, width=2),
                                     marker=dict(size=6, color=CYAN),
                                     name="polygon"))
if len(st.session_state.ruler_points) == 2:
    xs, ys = zip(*st.session_state.ruler_points)
    overlay_fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="ruler"))
if len(st.session_state.angle_points) == 3:
    xs, ys = zip(*st.session_state.angle_points)
    overlay_fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="angle"))

st.plotly_chart(overlay_fig, use_container_width=True, config={"displaylogo": False})

# Transform preview + export
st.header("Preview and Export")
params = asdict(st.session_state.params)
params["polygon_points"] = st.session_state.poly_points

if len(st.session_state.poly_points) >= 3:
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
    center = (float(xs.mean()), float(ys.mean())) if len(xs) else (W/2, H/2)

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
    st.caption("Draw a closed path and click 'Use last closed path' to preview transform.")
