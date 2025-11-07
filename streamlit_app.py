
import io, sys, subprocess

# ---- Self-healing install for the canvas component ----
try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "git+https://github.com/andfanilo/streamlit-drawable-canvas.git#egg=streamlit-drawable-canvas"])
        from streamlit_drawable_canvas import st_canvas  # retry
    except Exception as e:
        import streamlit as st
        st.error("Could not install streamlit-drawable-canvas automatically. "
                 "Please try again or use the fallback mode below.")
        st.stop()

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

st.set_page_config(page_title="Ilizarov 2D — Streamlit", layout="wide")

@dataclass
class Line2D:
    p1: Tuple[float,float]
    p2: Tuple[float,float]

def polygon_mask(size: Tuple[int,int], points: List[Tuple[float,float]]):
    mask = Image.new("L", size, 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=255, outline=255)
    return mask

def split_image_by_polygon(img: Image.Image, poly_pts: List[Tuple[float,float]]):
    mask_poly = polygon_mask(img.size, poly_pts)
    mask_inv = ImageOps.invert(mask_poly)
    outer = Image.new("RGBA", img.size, (0,0,0,0))
    inner = Image.new("RGBA", img.size, (0,0,0,0))
    outer.paste(img, (0,0), mask_inv)
    inner.paste(img, (0,0), mask_poly)
    return outer, inner, mask_inv, mask_poly

def center_of_mask(mask_img: Image.Image):
    arr = np.array(mask_img, dtype=float) / 255.0
    ys, xs = np.nonzero(arr > 0.5)
    if len(xs) == 0: return None
    return (float(xs.mean()), float(ys.mean()))

st.sidebar.title("Workflow")
st.sidebar.markdown("""
1. Upload image
2. Draw **lines** (proximal/distal), a **circle** (CORA), and a **polygon** (osteotomy) on the toolbar
3. Pick which is which in the selectors
4. Rotate the **distal** segment around CORA
5. Export PNG and CSV
""")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
canvas_w = st.sidebar.slider("Canvas width", 600, 1800, 1000, 10)
stroke_w = st.sidebar.slider("Stroke width", 1, 6, 2)
theta_deg = st.sidebar.slider("Rotate distal (deg)", -180, 180, 0, 1)

if not uploaded:
    st.info("Upload an image to begin."); st.stop()

base_img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGBA")
W, H = base_img.size
scale = canvas_w / W
canvas_h = int(H * scale)
display_img = base_img.resize((canvas_w, canvas_h), Image.BICUBIC)

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_color="#00FFFF",
    stroke_width=stroke_w,
    background_image=display_img,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode="transform",
    display_toolbar=True,
    key="main_canvas_autoinstall",
)

objs = canvas_result.json_data["objects"] if canvas_result.json_data and "objects" in canvas_result.json_data else []

def to_base(px, py): return (px/scale, py/scale)

lines: List[Line2D] = []
polys: List[List[Tuple[float,float]]] = []
circles: List[Tuple[float,float]] = []

for ob in objs:
    typ = ob.get("type")
    left = float(ob.get("left", 0)); top = float(ob.get("top", 0))
    if typ == "line":
        x1 = left; y1 = top
        x2 = left + float(ob.get("width", 0)); y2 = top + float(ob.get("height", 0))
        lines.append(Line2D(to_base(x1,y1), to_base(x2,y2)))
    elif typ == "polygon":
        pts = []
        for pt in ob.get("points", []):
            x = left + float(pt.get("x", 0)); y = top + float(pt.get("y", 0))
            pts.append(to_base(x, y))
        if len(pts) >= 3: polys.append(pts)
    elif typ == "circle":
        rx = float(ob.get("rx", ob.get("radius", 0))); ry = float(ob.get("ry", ob.get("radius", 0)))
        cx = left + rx; cy = top + ry
        circles.append(to_base(cx, cy))

st.header("Select elements")
proximal_line = distal_line = None
cora_pt = None
if lines:
    proximal_line = lines[st.selectbox("Proximal line", list(range(len(lines))), format_func=lambda i: f"Line {i+1}")]
if lines:
    distal_line = lines[st.selectbox("Distal line", list(range(len(lines))), index=min(1, len(lines)-1), format_func=lambda i: f"Line {i+1}")]
if circles:
    cora_pt = circles[st.selectbox("CORA (circle)", list(range(len(circles))), format_func=lambda i: f"Circle {i+1}")]
if polys:
    polygon_pts = polys[st.selectbox("Osteotomy polygon", list(range(len(polys))), format_func=lambda i: f"Polygon {i+1}")]
else:
    polygon_pts = []

if polygon_pts and cora_pt:
    outer, inner, mask_inv, mask_poly = split_image_by_polygon(base_img, polygon_pts)
    distal_piece = inner
    proximal_piece = outer

    moved = distal_piece.rotate(theta_deg, resample=Image.BICUBIC, center=cora_pt, expand=False)
    composed = Image.new("RGBA", base_img.size, (0,0,0,0))
    composed = Image.alpha_composite(composed, proximal_piece)
    composed = Image.alpha_composite(composed, moved)

    preview = composed.resize((canvas_w, canvas_h), Image.BICUBIC)
    st.image(preview, caption=f"Rotated distal around CORA by {theta_deg}°", use_container_width=True)

    import pandas as pd, io as _io
    params = dict(theta_deg=theta_deg, cora=cora_pt, polygon_points=polygon_pts)
    if proximal_line: params["proximal_line"] = [proximal_line.p1, proximal_line.p2]
    if distal_line: params["distal_line"] = [distal_line.p1, distal_line.p2]
    df = pd.DataFrame([params])
    st.download_button("Download parameters CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="ilizarov_params.csv", mime="text/csv")
    buf = _io.BytesIO(); composed.save(buf, format="PNG")
    st.download_button("Download transformed PNG", data=buf.getvalue(), file_name="ilizarov_transformed.png", mime="image/png")
else:
    st.info("Draw/select polygon and CORA to preview.")
