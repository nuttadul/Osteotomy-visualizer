
import importlib.util
import io
from typing import List, Tuple, Optional
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import os

st.set_page_config(page_title="Bone Ninja – Streamlit Adapter", layout="wide")

def load_engine(module_path="engine.py"):
    if not os.path.exists(module_path):
        return None
    spec = importlib.util.spec_from_file_location("engine", module_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
        if hasattr(mod, "apply_transform"):
            return mod
    except Exception as e:
        st.warning(f"Couldn't import engine.py: {e}")
    return None

ENGINE = load_engine()

def decode_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def polygon_mask(size, points):
    mask = Image.new("L", size, 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=255, outline=255)
    return mask

def apply_affine(img, dx, dy, rot_deg, center):
    rotated = img.rotate(rot_deg, resample=Image.BICUBIC, center=center, expand=False)
    canvas = Image.new("RGBA", img.size, (0,0,0,0))
    canvas.alpha_composite(rotated, (int(round(dx)), int(round(dy))))
    return canvas

def paste_with_mask(base, overlay, mask):
    out = base.copy()
    out.paste(overlay, (0,0), mask)
    return out

def centroid_of_polygon(pts):
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

ss = st.session_state
for k, v in dict(poly=[], cora=None, hinge=None, prox=[], dist=[], ruler=[], dispw=1100).items():
    ss.setdefault(k, v)

st.sidebar.title("Streamlit Adapter")
uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
tool = st.sidebar.radio("Tool", ["Polygon","CORA","HINGE","Proximal line","Distal line","Ruler"])
seg = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True)
center_mode = st.sidebar.radio("Center", ["HINGE","CORA","Polygon centroid"], index=0)
ss.dispw = st.sidebar.slider("Preview width", 600, 1800, ss.dispw, 50)
dx = st.sidebar.slider("ΔX", -1000, 1000, 0, 1)
dy = st.sidebar.slider("ΔY", -1000, 1000, 0, 1)
theta = st.sidebar.slider("Rotate (deg)", -180, 180, 0, 1)

cols = st.sidebar.columns(5)
if cols[0].button("Undo"):
    if tool == "Polygon" and ss.poly: ss.poly.pop()
    elif tool == "Proximal line" and ss.prox: ss.prox.pop()
    elif tool == "Distal line" and ss.dist: ss.dist.pop()
    elif tool == "Ruler" and ss.ruler: ss.ruler.pop()
    elif tool == "CORA": ss.cora=None
    elif tool == "HINGE": ss.hinge=None
if cols[1].button("Reset polygon"): ss.poly.clear()
if cols[2].button("Reset lines"): ss.prox.clear(); ss.dist.clear()
if cols[3].button("Clear centers"): ss.cora=None; ss.hinge=None
if cols[4].button("Clear ruler"): ss.ruler.clear()

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_img(data: bytes):
    img = decode_image(data)
    return img, img.size

img, (W,H) = load_img(uploaded.getvalue())
disp_w = min(ss.dispw, W)
scale = disp_w / float(W)
disp_h = int(round(H*scale))

preview = img.copy()
d = ImageDraw.Draw(preview)
if ss.poly:
    d.line(ss.poly, fill=(0,255,255,255), width=2)
    if len(ss.poly) >= 3:
        d.line([*ss.poly, ss.poly[0]], fill=(0,255,255,255), width=2)
if ss.prox:
    if len(ss.prox)==1: p=ss.prox[0]; d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=(66,133,244,255))
    elif len(ss.prox)==2: d.line(ss.prox, fill=(66,133,244,255), width=3)
if ss.dist:
    if len(ss.dist)==1: p=ss.dist[0]; d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=(221,0,221,255))
    elif len(ss.dist)==2: d.line(ss.dist, fill=(221,0,221,255), width=3)
if ss.ruler:
    if len(ss.ruler)==1: r=ss.ruler[0]; d.ellipse([r[0]-4,r[1]-4,r[0]+4,r[1]+4], fill=(255,0,0,255))
    elif len(ss.ruler)==2: d.line(ss.ruler, fill=(255,0,0,255), width=2)
if ss.cora:
    x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)
if ss.hinge:
    x,y=ss.hinge; d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
    d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
    d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)

disp_img = preview.resize((disp_w, disp_h), Image.NEAREST)
res = streamlit_image_coordinates(disp_img, key="click", width=disp_w, css={"cursor":"crosshair"})

def to_orig(pt): return (float(pt[0])/scale, float(pt[1])/scale)

if res and "x" in res and "y" in res:
    pt = to_orig((res["x"], res["y"]))
    if tool=="Polygon": ss.poly.append(pt)
    elif tool=="CORA": ss.cora=pt
    elif tool=="HINGE": ss.hinge=pt
    elif tool=="Proximal line":
        if len(ss.prox)>=2: ss.prox.clear()
        ss.prox.append(pt)
    elif tool=="Distal line":
        if len(ss.dist)>=2: ss.dist.clear()
        ss.dist.append(pt)
    elif tool=="Ruler":
        if len(ss.ruler)>=2: ss.ruler.clear()
        ss.ruler.append(pt)
    st.success(f"{tool} point: ({pt[0]:.1f},{pt[1]:.1f})")

center = None
if center_mode=="HINGE" and ss.hinge: center=ss.hinge
elif center_mode=="CORA" and ss.cora: center=ss.cora
else: center = centroid_of_polygon(ss.poly) or ss.cora or ss.hinge

st.header("Preview & Export")
if len(ss.poly)>=3 and center is not None:
    if ENGINE and hasattr(ENGINE, "apply_transform"):
        out_img = ENGINE.apply_transform(img, ss.poly, center, dx, dy, theta, seg)
    else:
        mask_poly = polygon_mask(img.size, ss.poly)
        mask_inv = ImageOps.invert(mask_poly)
        prox = Image.new("RGBA", img.size, (0,0,0,0))
        dist = Image.new("RGBA", img.size, (0,0,0,0))
        prox = paste_with_mask(prox, img, mask_inv)
        dist = paste_with_mask(dist, img, mask_poly)
        moving = dist if seg=="distal" else prox
        fixed = prox if seg=="distal" else dist
        moved = apply_affine(moving, dx=dx, dy=dy, rot_deg=theta, center=center)
        canvas = Image.new("RGBA", img.size, (0,0,0,0))
        out_img = Image.alpha_composite(Image.alpha_composite(canvas, fixed), moved)

    st.image(out_img.resize((disp_w, disp_h), Image.NEAREST), use_container_width=True)

    params = dict(mode=seg, dx=dx, dy=dy, rotate_deg=theta,
                  rotation_center=center, polygon_points=ss.poly,
                  cora=ss.cora, hinge=ss.hinge,
                  proximal_line=ss.prox, distal_line=ss.dist)
    df = pd.DataFrame([params])
    st.download_button("Download parameters CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="osteotomy_params.csv", mime="text/csv")
    buf = io.BytesIO(); out_img.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)", data=buf.getvalue(),
        file_name="osteotomy_transformed.png", mime="image/png")
else:
    st.info("Draw polygon (≥3) and pick a center (HINGE/CORA/centroid).")
