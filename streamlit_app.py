
import io
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

st.set_page_config(page_title="Bone Ninja — Safe Mode", layout="wide")

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

def parse_points(text: str) -> List[Tuple[float,float]]:
    pts = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            a, b = line.split(",", 1)
        elif "\t" in line:
            a, b = line.split("\t", 1)
        else:
            parts = line.split()
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
            else:
                continue
        try:
            pts.append((float(a), float(b)))
        except:
            continue
    return pts

def length_of_line(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))

def angle_from_three_points(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

st.title("Bone Ninja — Safe Mode (no extra components)")
st.write("Paste polygon and measurement points. This version avoids any custom widgets so it will run everywhere.")

uploaded = st.file_uploader("Upload image (png/jpg/tif)", type=["png","jpg","jpeg","tif","tiff"])
if not uploaded:
    st.stop()

img = pil_from_bytes(uploaded.getvalue())
W, H = img.size
st.image(img, caption=f"Original ({W}×{H})", use_container_width=True)

st.header("Polygon (osteotomy lasso)")
poly_text = st.text_area("Enter points (x,y) one per line", height=160, placeholder="120,340\n180,360\n210,420\n150,450")
poly_pts = parse_points(poly_text)
st.caption(f"{len(poly_pts)} points parsed")

col_m, col_t = st.columns(2)
with col_m:
    mode = st.radio("Segment to move", ["distal","proximal"], horizontal=True)
with col_t:
    dx = st.number_input("ΔX (px)", value=0.0, step=1.0)
    dy = st.number_input("ΔY (px)", value=0.0, step=1.0)
    rot = st.number_input("Rotate (deg)", value=0.0, step=1.0)

st.header("Measurements (optional)")
ruler_text = st.text_input("Ruler points (x1,y1 ; x2,y2)", value="")
angle_text = st.text_input("Angle points (x1,y1 ; x2,y2 ; x3,y3)", value="")
cal_mm = st.number_input("Calibration: known distance (mm) for current ruler", value=0.0, step=0.1)

# Parse measurements
def parse_inline_pairs(s: str) -> List[Tuple[float,float]]:
    s = s.strip()
    if not s:
        return []
    try:
        parts = [p.strip() for p in s.split(";")]
        def pp(x):
            a,b = x.split(",")
            return (float(a), float(b))
        return [pp(p) for p in parts if p]
    except Exception:
        return []

ruler_pts = parse_inline_pairs(ruler_text)
angle_pts = parse_inline_pairs(angle_text)

if len(ruler_pts) == 2:
    dpx = length_of_line(*ruler_pts)
    msg = f"Ruler: {dpx:.2f} px"
    if cal_mm > 0:
        px_per_mm = dpx / cal_mm
        msg += f"  ({cal_mm:.2f} mm → {px_per_mm:.3f} px/mm)"
    st.info(msg)

if len(angle_pts) == 3:
    ang = angle_from_three_points(angle_pts[0], angle_pts[1], angle_pts[2])
    st.info(f"Angle: {ang:.2f}°")

st.header("Preview and Export")
if len(poly_pts) < 3:
    st.warning("Enter at least 3 polygon points to preview the transform.")
    st.stop()

mask_poly = polygon_mask(img.size, poly_pts)
mask_inv = ImageOps.invert(mask_poly)

proximal_piece = Image.new("RGBA", img.size, (0,0,0,0))
distal_piece = Image.new("RGBA", img.size, (0,0,0,0))
proximal_piece = paste_with_mask(proximal_piece, img, mask_inv)
distal_piece = paste_with_mask(distal_piece, img, mask_poly)

seg_mask = mask_poly if mode == "distal" else mask_inv
arr = np.array(seg_mask) / 255.0
ys, xs = np.nonzero(arr > 0.5)
center = (float(xs.mean()), float(ys.mean())) if len(xs) else (W/2, H/2)

moving = distal_piece if mode == "distal" else proximal_piece
fixed = proximal_piece if mode == "distal" else distal_piece

moved = apply_affine(moving, dx=dx, dy=dy, rot_deg=rot, center=center)

composed = Image.new("RGBA", img.size, (0,0,0,0))
composed = Image.alpha_composite(composed, fixed)
composed = Image.alpha_composite(composed, moved)

st.image(composed, caption=f"Transformed ({mode} moved)", use_container_width=True)

# Export
import pandas as pd
params = asdict(TransformParams(mode, dx, dy, rot))
params["polygon_points"] = poly_pts
df = pd.DataFrame([params])
st.download_button("Download parameters CSV", data=df.to_csv(index=False).encode("utf-8"),
                   file_name="osteotomy_params.csv", mime="text/csv")
buf = io.BytesIO()
composed.save(buf, format="PNG")
st.download_button("Download transformed image (PNG)", data=buf.getvalue(),
                   file_name="osteotomy_transformed.png", mime="image/png")
