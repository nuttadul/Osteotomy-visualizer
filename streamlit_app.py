
import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd

st.set_page_config(page_title="Bone Ninja — Auto‑Click", layout="wide")

CYAN = "cyan"

@dataclass
class TransformParams:
    mode: str  # "distal" or "proximal"
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

def length_of_line(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))

# ---------------- State ----------------
ss = st.session_state
if "poly_points" not in ss: ss.poly_points: List[Tuple[float,float]] = []
if "ruler_points" not in ss: ss.ruler_points: List[Tuple[float,float]] = []
if "cora_pt" not in ss: ss.cora_pt = None
if "click_count" not in ss: ss.click_count = 0

# ---------------- Sidebar ----------------
st.sidebar.title("Workflow")
st.sidebar.markdown("""
1. Upload image
2. Pick a tool (Polygon / CORA / Ruler)
3. **Just click on the image** — points are auto-added
4. Adjust ΔX / ΔY / Rotate
5. Export
""")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Tool", ["Polygon","CORA","Ruler"], horizontal=True)
segment_choice = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True)

st.sidebar.subheader("Translation")
dx = st.sidebar.slider("ΔX (px)", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY (px)", -1000, 1000, 0, 1)
st.sidebar.subheader("Rotation")
rotate_deg = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)

colB1, colB2, colB3 = st.sidebar.columns(3)
with colB1:
    if st.button("Undo"):
        if tool == "Polygon" and ss.poly_points: ss.poly_points.pop()
        elif tool == "Ruler" and ss.ruler_points: ss.ruler_points.pop()
        elif tool == "CORA": ss.cora_pt = None
with colB2:
    if st.button("Clear all"):
        ss.poly_points.clear(); ss.ruler_points.clear(); ss.cora_pt=None
with colB3:
    if st.button("Close polygon"):
        pass  # visual closure handled by drawing when >=3 points

if uploaded is None:
    st.info("Upload an image to begin."); st.stop()

base_img = pil_from_bytes(uploaded.getvalue())
W, H = base_img.size

# ---------------- Figure ----------------
fig = go.Figure()
fig.update_xaxes(range=[0, W], constrain="domain", visible=False)
fig.update_yaxes(range=[H, 0], scaleanchor="x", scaleratio=1, visible=False)
fig.add_layout_image(dict(source=base_img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", layer="below"))
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), dragmode="pan", clickmode="event+select")

# Draw overlays so far
if ss.poly_points:
    xs, ys = zip(*ss.poly_points)
    fig.add_trace(go.Scatter(x=list(xs)+([xs[0]] if len(xs)>=3 else []),
                             y=list(ys)+([ys[0]] if len(ys)>=3 else []),
                             mode="lines+markers",
                             line=dict(color=CYAN, width=2),
                             marker=dict(size=6, color=CYAN),
                             name="polygon"))
if ss.ruler_points:
    xs, ys = zip(*ss.ruler_points)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="ruler"))
if ss.cora_pt:
    fig.add_trace(go.Scatter(x=[ss.cora_pt[0]], y=[ss.cora_pt[1]], mode="markers",
                             marker=dict(size=10, symbol="circle"), name="CORA"))

st.caption("Click on the image. New clicks are automatically added to the active tool.")
events = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="plot_autoclick")

# Auto-append any new clicks
if events and len(events) > ss.click_count:
    new = events[ss.click_count:]  # process only new ones
    for ev in new:
        if "x" in ev and "y" in ev:
            pt = (float(ev["x"]), float(ev["y"]))
            if tool == "Polygon":
                ss.poly_points.append(pt)
            elif tool == "CORA":
                ss.cora_pt = pt
            elif tool == "Ruler":
                if len(ss.ruler_points) >= 2:
                    ss.ruler_points.clear()
                ss.ruler_points.append(pt)
    ss.click_count = len(events)

# Measurements
st.subheader("Measurements")
msg = ""
if len(ss.ruler_points) == 2:
    px = length_of_line(*ss.ruler_points)
    msg += f"Ruler: {px:.2f} px"
if msg: st.info(msg)
else: st.caption("Add a 2‑point ruler using the Ruler tool.")

# Transform preview + export
st.header("Preview and Export")
if ss.poly_points and ss.cora_pt:
    poly_pts = ss.poly_points
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

    params = dict(mode=segment_choice, dx=dx, dy=dy, rotate_deg=rotate_deg,
                  polygon_points=poly_pts, cora=ss.cora_pt)
    df_params = pd.DataFrame([params])
    st.download_button("Download parameters CSV", data=df_params.to_csv(index=False).encode("utf-8"),
                       file_name="osteotomy_params.csv", mime="text/csv")

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)", data=buf.getvalue(), file_name="osteotomy_transformed.png", mime="image/png")
else:
    st.info("Add polygon (≥3 points) and CORA by clicking on the image.")
