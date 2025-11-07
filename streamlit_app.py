
import io
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st

# Try to import the drawable canvas; if not present, show a clear error.
try:
    from streamlit_drawable_canvas import st_canvas
except Exception as e:
    st.error("streamlit-drawable-canvas is not installed. Please use the provided requirements.txt (GitHub install).")
    st.stop()

st.set_page_config(page_title="Ilizarov 2D — Streamlit", layout="wide")

# ---------------- Data structures ----------------
@dataclass
class Line2D:
    p1: Tuple[float,float]
    p2: Tuple[float,float]

    def as_array(self):
        return (np.array(self.p1, dtype=float), np.array(self.p2, dtype=float))

def line_dir(P1, P2):
    v = P2 - P1
    n = np.linalg.norm(v) + 1e-12
    return v / n

def rotate_point(P, H, theta_rad):
    px, py = P
    hx, hy = H
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    x = c*(px-hx) - s*(py-hy) + hx
    y = s*(px-hx) + c*(py-hy) + hy
    return (x, y)

def polygon_mask(size: Tuple[int,int], points: List[Tuple[float,float]]) -> Image.Image:
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
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))

# ---------------- Sidebar workflow ----------------
st.sidebar.title("Workflow")
st.sidebar.markdown("""
1. Upload X-ray/photo
2. Calibrate px↔mm (optional) using a line
3. Place **Proximal line** and **Distal line** (yellow) with the Line tool
4. Place **CORA** using a small circle or by entering coordinates
5. Draw **Osteotomy polygon** with the Polygon tool
6. Rotate the distal segment around CORA
""")

uploaded = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
canvas_w = st.sidebar.slider("Canvas width (px)", 600, 1800, 1000, 10)
stroke_w = st.sidebar.slider("Stroke width", 1, 6, 2)
px_per_mm = st.sidebar.number_input("Calibration px/mm (optional)", min_value=0.0, value=0.0, step=0.01,
                                    help="Enter directly or draw a calibration line then click 'Use last line for calibration'.")

theta_deg = st.sidebar.slider("Rotate distal (deg)", -180, 180, 0, 1, help="Rotation around CORA")
reset_btn = st.sidebar.button("Reset rotation")

# ---------------- State ----------------
if "theta_deg" not in st.session_state:
    st.session_state.theta_deg = 0
if reset_btn:
    st.session_state.theta_deg = 0
else:
    st.session_state.theta_deg = theta_deg

# ---------------- Main layout ----------------
colL, colR = st.columns([1.3, 1])
with colL:
    st.header("Canvas")
    if not uploaded:
        st.info("Upload an image to begin.")
        st.stop()

    # Prepare image and scaled canvas size (keep aspect)
    base_img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGBA")
    W, H = base_img.size
    scale = canvas_w / W
    canvas_h = int(H * scale)
    display_img = base_img.resize((canvas_w, canvas_h), Image.BICUBIC)

    # Canvas with drawing tools
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_color="#00FFFF",
        stroke_width=stroke_w,
        background_image=display_img,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="transform",  # we'll allow toolbox; user picks line/polygon/circle
        display_toolbar=True,
        key="main_canvas",
    )

with colR:
    st.header("Preview")
    # parse objects
    objs = canvas_result.json_data["objects"] if canvas_result.json_data and "objects" in canvas_result.json_data else []

    def to_base(px, py):
        return (px/scale, py/scale)

    proximal_line: Optional[Line2D] = None
    distal_line: Optional[Line2D] = None
    cora_pt: Optional[Tuple[float,float]] = None
    polygon_pts: List[Tuple[float,float]] = []
    calib_line: Optional[Line2D] = None

    # Heuristic mapping based on object order and colors: user can label via checklist below too
    # We'll collect all lines and polygons and let the controls choose which one is which.
    lines: List[Line2D] = []
    polys: List[List[Tuple[float,float]]] = []
    circles: List[Tuple[float,float]] = []

    for ob in objs:
        typ = ob.get("type")
        left = float(ob.get("left", 0))
        top  = float(ob.get("top", 0))
        if typ == "line":
            x1 = left; y1 = top
            x2 = left + float(ob.get("width", 0))
            y2 = top + float(ob.get("height", 0))
            p1 = to_base(x1, y1); p2 = to_base(x2, y2)
            lines.append(Line2D(p1, p2))
        elif typ == "polygon":
            pts = []
            for pt in ob.get("points", []):
                x = left + float(pt.get("x", 0))
                y = top  + float(pt.get("y", 0))
                pts.append(to_base(x, y))
            if len(pts) >= 3:
                polys.append(pts)
        elif typ == "circle":
            # approximate center from left/top + radius
            rx = float(ob.get("rx", ob.get("radius", 0)))
            ry = float(ob.get("ry", ob.get("radius", 0)))
            cx = left + rx
            cy = top + ry
            circles.append(to_base(cx, cy))

    # Selection controls
    if lines:
        idx_prox = st.selectbox("Choose proximal line", list(range(len(lines))), format_func=lambda i: f"Line {i+1}")
        proximal_line = lines[idx_prox]
    if lines:
        idx_dist = st.selectbox("Choose distal line (yellow)", list(range(len(lines))), index=min(1, len(lines)-1),
                                format_func=lambda i: f"Line {i+1}")
        distal_line = lines[idx_dist]
    if lines:
        idx_cal = st.selectbox("Choose calibration line (optional)", ["None"] + [f"Line {i+1}" for i in range(len(lines))])
        if idx_cal != "None":
            calib_line = lines[int(idx_cal.split()[-1]) - 1]
            if px_per_mm == 0.0:
                # try to set px/mm from this line if user provided mm below
                pass

    if circles:
        idx_cora = st.selectbox("Choose CORA (from circles)", list(range(len(circles))), format_func=lambda i: f"Circle {i+1}")
        cora_pt = circles[idx_cora]

    # If no circle, allow manual CORA entry
    if not cora_pt:
        with st.expander("Set CORA manually"):
            x = st.number_input("CORA x (px)", min_value=0.0, max_value=float(W), value=float(W/2))
            y = st.number_input("CORA y (px)", min_value=0.0, max_value=float(H), value=float(H/2))
            cora_pt = (x, y)

    # Osteotomy polygon
    if polys:
        idx_poly = st.selectbox("Choose osteotomy polygon", list(range(len(polys))), format_func=lambda i: f"Polygon {i+1}")
        polygon_pts = polys[idx_poly]

    # When we have polygon and lines, build the transform preview
    if polygon_pts:
        outer, inner, mask_inv, mask_poly = split_image_by_polygon(base_img, polygon_pts)

        # Which segment is distal? We interpret the **inside polygon** as distal cut-piece
        # but original behavior rotates distal segment + cut piece around hinge.
        # We'll treat inner as cut-piece (distal). Proximal is outer.
        distal_piece = inner
        proximal_piece = outer

        # Rotation center
        Hx, Hy = cora_pt

        # Rotate distal piece around CORA by theta
        theta = math.radians(st.session_state.theta_deg)
        # To rotate around CORA, we can shift by (0,0) and use rotate(center=(Hx,Hy))
        moved = distal_piece.rotate(st.session_state.theta_deg, resample=Image.BICUBIC, center=(Hx, Hy), expand=False)

        composed = Image.new("RGBA", base_img.size, (0,0,0,0))
        composed = Image.alpha_composite(composed, proximal_piece)
        composed = Image.alpha_composite(composed, moved)

        # Downscale to display size
        preview = composed.resize((canvas_w, canvas_h), Image.BICUBIC)
        st.image(preview, caption=f"Rotated distal around CORA by {st.session_state.theta_deg}°", use_container_width=True)

        # Exports
        import pandas as pd
        params = dict(theta_deg=st.session_state.theta_deg, cora=cora_pt, polygon_points=polygon_pts)
        if proximal_line: params["proximal_line"] = [tuple(map(float, proximal_line.p1)), tuple(map(float, proximal_line.p2))]
        if distal_line: params["distal_line"] = [tuple(map(float, distal_line.p1)), tuple(map(float, distal_line.p2))]
        if px_per_mm: params["px_per_mm"] = float(px_per_mm)

        df = pd.DataFrame([params])
        st.download_button("Download parameters CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="ilizarov_params.csv", mime="text/csv")
        buf = io.BytesIO()
        composed.save(buf, format="PNG")
        st.download_button("Download transformed PNG", data=buf.getvalue(), file_name="ilizarov_transformed.png", mime="image/png")

    else:
        st.info("Draw and select an osteotomy polygon to preview rotation.")

st.caption("Tip: Use the toolbar Line, Circle, and Polygon tools. Distal line is conceptual (yellow in original); here you select which line acts as distal.")
