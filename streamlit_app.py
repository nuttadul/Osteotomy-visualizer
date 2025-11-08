import io
from typing import List, Tuple, Optional
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy (Streamlit)", layout="wide")

# ---------- helpers ----------
def decode_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def polygon_mask(size, points: List[Tuple[float,float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(points) >= 3:
        ImageDraw.Draw(m).polygon(points, fill=255, outline=255)
    return m

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

def apply_affine(img: Image.Image, dx, dy, rot_deg, center_xy):
    rot = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", img.size, (0,0,0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

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

# ---------- UI & state ----------
ss = st.session_state
for k, v in dict(poly=[], cora=None, hinge=None, prox=[], dist=[], ruler=[],
                 dispw=1100, dx=0, dy=0, theta=0, segment="distal").items():
    ss.setdefault(k, v)

st.sidebar.header("Upload image")
uploaded = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Tool", ["Polygon","CORA","HINGE","Prox line","Dist line","Ruler"], index=0)
segment = st.sidebar.radio("Move segment", ["distal","proximal"], index=(0 if ss.segment=="distal" else 1), horizontal=True, key="segment")
ss.dispw = st.sidebar.slider("Preview width", 600, 1800, ss.dispw, 50)
ss.dx    = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy    = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)

c1, c2, c3, c4, c5 = st.sidebar.columns(5)
if c1.button("Undo"):
    if tool == "Polygon" and ss.poly: ss.poly.pop()
    elif tool == "Prox line" and ss.prox: ss.prox.pop()
    elif tool == "Dist line" and ss.dist: ss.dist.pop()
    elif tool == "Ruler" and ss.ruler: ss.ruler.pop()
    elif tool == "CORA": ss.cora = None
    elif tool == "HINGE": ss.hinge = None
if c2.button("Reset poly"): ss.poly.clear()
if c3.button("Reset lines"): ss.prox.clear(); ss.dist.clear()
if c4.button("Clear centers"): ss.cora=None; ss.hinge=None
if c5.button("Reset move"): ss.dx=0; ss.dy=0; ss.theta=0

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

img = decode_image(uploaded.getvalue())
W, H = img.size
disp_w = min(ss.dispw, W)
scale = disp_w / float(W)
disp_h = int(round(H*scale))

# crosshair cursor for better pointing
st.markdown("<style>img{cursor: crosshair !important;}</style>", unsafe_allow_html=True)

# paint overlays for preview
preview = img.copy()
d = ImageDraw.Draw(preview)
if ss.poly:
    d.line(ss.poly, fill=(0,255,255,255), width=2)
    if len(ss.poly) >= 3:
        d.line([*ss.poly, ss.poly[0]], fill=(0,255,255,255), width=2)
if len(ss.prox) == 2: d.line(ss.prox, fill=(66,133,244,255), width=3)
if len(ss.dist) == 2: d.line(ss.dist, fill=(221,0,221,255), width=3)
if ss.cora:
    x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)
if ss.hinge:
    x,y=ss.hinge; d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
    d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
    d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)

disp_img = preview.resize((disp_w, disp_h), Image.NEAREST)
res = streamlit_image_coordinates(disp_img, width=disp_w, key="clicks")

def to_orig(pt_disp): return (float(pt_disp[0]) / scale, float(pt_disp[1]) / scale)

if res and "x" in res and "y" in res:
    pt = to_orig((res["x"], res["y"]))
    if tool == "Polygon": ss.poly.append(pt)
    elif tool == "CORA":   ss.cora = pt
    elif tool == "HINGE":  ss.hinge = pt
    elif tool == "Prox line":
        if len(ss.prox) >= 2: ss.prox.clear()
        ss.prox.append(pt)
    elif tool == "Dist line":
        if len(ss.dist) >= 2: ss.dist.clear()
        ss.dist.append(pt)
    elif tool == "Ruler":
        if len(ss.ruler) >= 2: ss.ruler.clear()
        ss.ruler.append(pt)

# ---------- transform preview ----------
center = ss.hinge or ss.cora or centroid(ss.poly)
if len(ss.poly) >= 3 and center is not None:
    m = polygon_mask(img.size, ss.poly)
    inv = ImageOps.invert(m)
    prox = Image.new("RGBA", img.size, (0,0,0,0)); prox.paste(img, (0,0), inv)
    dist = Image.new("RGBA", img.size, (0,0,0,0)); dist.paste(img, (0,0), m)
    moving = dist if ss.segment=="distal" else prox
    fixed  = prox if ss.segment=="distal" else dist

    moved = apply_affine(moving, ss.dx, ss.dy, ss.theta, center)
    out = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", img.size, (0,0,0,0)), fixed), moved)

    # redraw lines with following
    draw2 = ImageDraw.Draw(out)
    if len(ss.dist) == 2:
        p = transform_points(ss.dist, ss.dx, ss.dy, ss.theta, center) if ss.segment=="distal" else ss.dist
        draw2.line(p, fill=(221,0,221,255), width=3)
    if len(ss.prox) == 2:
        p = transform_points(ss.prox, ss.dx, ss.dy, ss.theta, center) if ss.segment=="proximal" else ss.prox
        draw2.line(p, fill=(66,133,244,255), width=3)

    st.image(out.resize((disp_w, disp_h), Image.NEAREST), use_container_width=True)

    params = dict(mode=ss.segment, dx=ss.dx, dy=ss.dy, rotate_deg=ss.theta,
                  rotation_center=center, polygon_points=ss.poly,
                  cora=ss.cora, hinge=ss.hinge,
                  proximal_line=ss.prox, distal_line=ss.dist)
    df = pd.DataFrame([params])
    st.download_button("Download parameters CSV",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="osteotomy_params.csv", mime="text/csv")
    buf = io.BytesIO(); out.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)", data=buf.getvalue(),
                       file_name="osteotomy_transformed.png", mime="image/png")
else:
    st.info("Draw polygon (≥3) and set HINGE/CORA; clicking is active.")
