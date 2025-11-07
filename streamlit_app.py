
import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

st.set_page_config(page_title="Bone Ninja — Simple Clicks", layout="wide")

CYAN = (0, 255, 255, 255)

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

# ---------------- Sidebar ----------------
st.sidebar.title("Workflow")
st.sidebar.markdown("""
Click on the image to add a point to the active tool.
No Plotly tools. Clicks always register.
""")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Active tool", ["Polygon","CORA","Ruler"], horizontal=True, index=0)
segment_choice = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True, index=0)

st.sidebar.subheader("Translation")
dx = st.sidebar.slider("ΔX (px)", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY (px)", -1000, 1000, 0, 1)
st.sidebar.subheader("Rotation")
rotate_deg = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)

b1, b2, b3 = st.sidebar.columns(3)
with b1:
    if st.button("Undo"):
        if tool == "Polygon" and ss.poly_points: ss.poly_points.pop()
        elif tool == "Ruler" and ss.ruler_points: ss.ruler_points.pop()
        elif tool == "CORA": ss.cora_pt = None
with b2:
    if st.button("Clear all"):
        ss.poly_points.clear(); ss.ruler_points.clear(); ss.cora_pt=None
with b3:
    st.write("")

if uploaded is None:
    st.info("Upload an image to begin."); st.stop()

base_img = pil_from_bytes(uploaded.getvalue())
W, H = base_img.size

# --------- Draw current overlays for preview (this is also the clickable image) ---------
preview = base_img.copy()
draw = ImageDraw.Draw(preview)

# polygon
if len(ss.poly_points) >= 1:
    draw.line(ss.poly_points, fill=CYAN, width=2, joint="curve")
    if len(ss.poly_points) >= 3:
        draw.line([*ss.poly_points, ss.poly_points[0]], fill=CYAN, width=2)

# ruler
if len(ss.ruler_points) == 1:
    r1 = ss.ruler_points[0]
    draw.ellipse([r1[0]-3, r1[1]-3, r1[0]+3, r1[1]+3], fill=(255,0,0,255))
elif len(ss.ruler_points) == 2:
    draw.line(ss.ruler_points, fill=(255,0,0,255), width=2)
    for r in ss.ruler_points:
        draw.ellipse([r[0]-3, r[1]-3, r[0]+3, r[1]+3], fill=(255,0,0,255))

# cora
if ss.cora_pt:
    x,y = ss.cora_pt
    draw.ellipse([x-5,y-5,x+5,y+5], outline=(0,255,0,255), width=2)

st.caption("Click on the image below. Your click is immediately added to the active tool.")
res = streamlit_image_coordinates(preview, key="imgclick", width=min(1000, W))

if res is not None and "x" in res and "y" in res:
    pt = (float(res["x"]), float(res["y"]))
    if tool == "Polygon":
        ss.poly_points.append(pt)
    elif tool == "CORA":
        ss.cora_pt = pt
    elif tool == "Ruler":
        if len(ss.ruler_points) >= 2: ss.ruler_points.clear()
        ss.ruler_points.append(pt)
    st.success(f"Added point x={pt[0]:.1f}, y={pt[1]:.1f} to {tool}")

# Measurements
st.subheader("Measurements")
if len(ss.ruler_points) == 2:
    st.info(f"Ruler: {length_of_line(*ss.ruler_points):.2f} px")
else:
    st.caption("Use Ruler tool and click two points.")

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
    out_img = Image.alpha_composite(Image.alpha_composite(composed, fixed), moved)
    st.image(out_img, caption=f"Transformed ({segment_choice} moved)", use_container_width=True)

    params = dict(mode=segment_choice, dx=dx, dy=dy, rotate_deg=rotate_deg,
                  polygon_points=poly_pts, cora=ss.cora_pt)
    df = pd.DataFrame([params])
    st.download_button("Download parameters CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="osteotomy_params.csv", mime="text/csv")

    buf = io.BytesIO(); out_img.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)", data=buf.getvalue(),
                       file_name="osteotomy_transformed.png", mime="image/png")
else:
    st.info("Add polygon points (≥3) and set CORA to preview transform.")
