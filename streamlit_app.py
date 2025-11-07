
import io
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd

st.set_page_config(page_title="Bone Ninja (Streamlit Edition)", layout="wide")

# ---------------------------
# Helpers
# ---------------------------

CYAN = "#00FFFF"

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
    """Create a mask image (L) with 255 inside polygon."""
    mask = Image.new("L", size, 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=255, outline=255)
    return mask

def apply_affine(img: Image.Image, dx: float, dy: float, rot_deg: float, center: Tuple[float,float]) -> Image.Image:
    """Rotate around center, then translate. Returns an RGBA image transformed on transparent canvas."""
    # PIL rotate is around center if expand=False when using Image.rotate with center parameter (Pillow>=8.0)
    # We'll rotate first, then paste with translation (dx, dy)
    rotated = img.rotate(rot_deg, resample=Image.BICUBIC, center=center, expand=False)
    # After rotation, perform translation by shifting the image on a transparent canvas
    canvas = Image.new("RGBA", img.size, (0,0,0,0))
    # Translation is applied by offsetting when pasting
    canvas.alpha_composite(rotated, (int(round(dx)), int(round(dy))))
    return canvas

def paste_with_mask(base: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
    out = base.copy()
    out.paste(overlay, (0,0), mask)
    return out

def angle_from_three_points(a: Tuple[float,float], b: Tuple[float,float], c: Tuple[float,float]) -> float:
    """Return angle ABC in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def length_of_line(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))

def transparent_background(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

def draw_points(draw: ImageDraw.ImageDraw, points: List[Tuple[float,float]], color: str):
    r = 2
    for x, y in points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color)

# ---------------------------
# Sidebar Controls
# ---------------------------

st.sidebar.title("Workflow")
st.sidebar.markdown("""
1. Upload image
2. Select a tool
3. Draw on canvas
4. Adjust translation/rotation for the selected segment
5. Export results
""")

uploaded = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"])
canvas_width = st.sidebar.slider("Canvas width", min_value=512, max_value=1600, value=900, step=10)
stroke_width = st.sidebar.slider("Drawing stroke width", 1, 8, 2)
stroke_color = st.sidebar.color_picker("Stroke color (drawing phase)", CYAN, key="stroke_color")

tool = st.sidebar.radio("Tool", ["Osteotomy (lasso polygon)", "Ruler (2 points)", "Angle (3 points)"])

segment_choice = st.sidebar.radio("Segment to move", ["distal", "proximal"], horizontal=True)

# Sliders organized in two groups
st.sidebar.subheader("Translation")
dx = st.sidebar.slider("ΔX (pixels)", min_value=-1000, max_value=1000, value=0, step=1)
dy = st.sidebar.slider("ΔY (pixels)", min_value=-1000, max_value=1000, value=0, step=1)

st.sidebar.subheader("Rotation")
rotate_deg = st.sidebar.slider("Rotate (deg)", min_value=-180, max_value=180, value=0, step=1)

reset = st.sidebar.button("Reset Transformations")

st.sidebar.subheader("Calibration (optional)")
cal_mm = st.sidebar.number_input("Known distance (mm) for last ruler line", min_value=0.0, value=0.0, step=0.1, help="Set this after you draw a line with the Ruler tool.")
apply_cal = st.sidebar.button("Apply calibration to last ruler")

# ---------------------------
# State
# ---------------------------

if "params" not in st.session_state:
    st.session_state.params = TransformParams(mode="distal", dx=0.0, dy=0.0, rotate_deg=0.0)

if "measurements" not in st.session_state:
    st.session_state.measurements: List[MeasurementRecord] = []

if reset:
    st.session_state.params = TransformParams(mode=segment_choice, dx=0.0, dy=0.0, rotate_deg=0.0)

# Keep mode updated
st.session_state.params.mode = segment_choice
st.session_state.params.dx = dx
st.session_state.params.dy = dy
st.session_state.params.rotate_deg = rotate_deg

# Calibration state
if "px_per_mm" not in st.session_state:
    st.session_state.px_per_mm = None  # float

# ---------------------------
# Main Layout
# ---------------------------

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.header("Canvas")
    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()

    base_img = pil_from_bytes(uploaded.getvalue())
    # Fit height to maintain aspect
    w, h = base_img.size
    scale = canvas_width / w
    canvas_height = int(h * scale)

    # Build a background image for the canvas (downscaled for drawing)
    display_img = base_img.resize((canvas_width, canvas_height), Image.BICUBIC)

    # Determine drawing mode
    drawing_mode = "polygon" if tool.startswith("Osteotomy") else "freedraw"  # we will post-process points
    if tool == "Ruler (2 points)":
        drawing_mode = "transform"  # we use rect/line approximate; but st_canvas only supports a limited set; we'll parse from JSON

    if tool == "Angle (3 points)":
        drawing_mode = "transform"

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=display_img,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon" if tool.startswith("Osteotomy") else "transform",
        initial_drawing=None,
        key=f"canvas_{tool}",
        display_toolbar=True,
    )

with col_right:
    st.header("Preview and Export")

    # Parse points from canvas JSON
    polygon_points_display: List[Tuple[float,float]] = []
    poly_ok = False
    line_points: List[Tuple[float,float]] = []
    angle_points: List[Tuple[float,float]] = []

    # st_canvas returns a JSON-like object in .json_data; shapes contain normalized coords.
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objs = canvas_result.json_data["objects"]

        # Helper to scale back to original image coordinates
        def to_base_coords(px: float, py: float) -> Tuple[float,float]:
            return (px / scale, py / scale)

        if tool.startswith("Osteotomy"):
            # Find the last polygon
            for ob in objs:
                if ob.get("type") == "polygon":
                    # Extract points relative to the object position
                    left = float(ob.get("left", 0))
                    top = float(ob.get("top", 0))
                    points = ob.get("points", [])
                    pts = []
                    for pt in points:
                        x = left + float(pt.get("x", 0))
                        y = top + float(pt.get("y", 0))
                        pts.append(to_base_coords(x, y))
                    polygon_points_display = pts
            poly_ok = len(polygon_points_display) >= 3

        elif tool.startswith("Ruler"):
            # Use two circle/rect/transform handles as points
            # We'll try to find two 'active' points by reading path or line objects
            pts2 = []
            for ob in objs:
                if ob.get("type") in ("circle", "rect", "triangle", "line"):
                    x = float(ob.get("left", 0)) + float(ob.get("width", 0))/2
                    y = float(ob.get("top", 0)) + float(ob.get("height", 0))/2
                    pts2.append(to_base_coords(x, y))
                if ob.get("type") == "path":
                    # Use first and last point of the path
                    path = ob.get("path", [])
                    if len(path) >= 2:
                        x1 = path[0][1]; y1 = path[0][2]
                        x2 = path[-1][1]; y2 = path[-1][2]
                        pts2.append(to_base_coords(x1, y1))
                        pts2.append(to_base_coords(x2, y2))
            if len(pts2) >= 2:
                line_points = [pts2[0], pts2[1]]
                # Save a measurement if exactly two points
                length_px = length_of_line(*line_points)
                record = MeasurementRecord(
                    timestamp=time.time(),
                    tool="ruler",
                    values={"p1": line_points[0], "p2": line_points[1], "length_px": length_px}
                )
                # Store only the last measurement (avoid flooding)
                if len(st.session_state.measurements) == 0 or st.session_state.measurements[-1].tool != "ruler":
                    st.session_state.measurements.append(record)
                else:
                    st.session_state.measurements[-1] = record

        elif tool.startswith("Angle"):
            # Aim to get 3 points
            pts3 = []
            for ob in objs:
                if ob.get("type") in ("circle", "rect", "triangle", "line"):
                    x = float(ob.get("left", 0)) + float(ob.get("width", 0))/2
                    y = float(ob.get("top", 0)) + float(ob.get("height", 0))/2
                    pts3.append(to_base_coords(x, y))
                if ob.get("type") == "path":
                    path = ob.get("path", [])
                    if len(path) >= 3:
                        x1 = path[0][1]; y1 = path[0][2]
                        x2 = path[len(path)//2][1]; y2 = path[len(path)//2][2]
                        x3 = path[-1][1]; y3 = path[-1][2]
                        pts3.append(to_base_coords(x1, y1))
                        pts3.append(to_base_coords(x2, y2))
                        pts3.append(to_base_coords(x3, y3))
            if len(pts3) >= 3:
                angle_points = pts3[:3]
                ang = angle_from_three_points(angle_points[0], angle_points[1], angle_points[2])
                record = MeasurementRecord(
                    timestamp=time.time(),
                    tool="angle",
                    values={"a": angle_points[0], "b": angle_points[1], "c": angle_points[2], "angle_deg": ang}
                )
                if len(st.session_state.measurements) == 0 or st.session_state.measurements[-1].tool != "angle":
                    st.session_state.measurements.append(record)
                else:
                    st.session_state.measurements[-1] = record

    # Measurements display
    st.subheader("Measurements")
    if st.session_state.measurements:
        df_meas = pd.DataFrame([
            {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp)),
             "tool": m.tool, **m.values}
            for m in st.session_state.measurements
        ])
        st.dataframe(df_meas, use_container_width=True)
    else:
        st.caption("No measurements yet. Use Ruler or Angle tool.")

    # Calibration logic
    if apply_cal and cal_mm > 0 and st.session_state.measurements:
        # Find last ruler
        last = None
        for m in reversed(st.session_state.measurements):
            if m.tool == "ruler":
                last = m
                break
        if last is not None:
            px = float(last.values.get("length_px", 0.0))
            if px > 0:
                st.session_state.px_per_mm = px / cal_mm
                st.success(f"Calibration set: {st.session_state.px_per_mm:.3f} px/mm")

    if st.session_state.px_per_mm:
        st.caption(f"Current calibration: {st.session_state.px_per_mm:.3f} px/mm")

    # Report current ruler length with mm if available
    if line_points:
        px_len = length_of_line(*line_points)
        msg = f"Ruler length: {px_len:.2f} px"
        if st.session_state.px_per_mm:
            mm = px_len / st.session_state.px_per_mm
            msg += f"  ({mm:.2f} mm)"
        st.info(msg)

    if angle_points:
        ang = angle_from_three_points(angle_points[0], angle_points[1], angle_points[2])
        st.info(f"Angle: {ang:.2f}°")

    # ---------------------------
    # Apply Transformations if polygon exists
    # ---------------------------
    transformed_preview = None
    if tool.startswith("Osteotomy") and poly_ok:
        # Prepare full-res polygon points
        poly_pts = [(float(x), float(y)) for x, y in polygon_points_display]

        # Build masks for proximal and distal pieces
        mask_poly = polygon_mask(base_img.size, poly_pts)
        mask_inv = ImageOps.invert(mask_poly)

        # Split image
        proximal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
        distal_piece = Image.new("RGBA", base_img.size, (0,0,0,0))
        proximal_piece = paste_with_mask(proximal_piece, base_img, mask_inv)  # outside polygon
        distal_piece = paste_with_mask(distal_piece, base_img, mask_poly)     # inside polygon

        # Determine centroid to use as rotation center (of the selected segment)
        if segment_choice == "distal":
            seg_mask = mask_poly
        else:
            seg_mask = mask_inv

        # Compute center of mass (approximate) from mask
        mask_arr = np.array(seg_mask) / 255.0
        ys, xs = np.nonzero(mask_arr > 0.5)
        if len(xs) == 0:
            center = (base_img.size[0]//2, base_img.size[1]//2)
        else:
            cx = float(xs.mean())
            cy = float(ys.mean())
            center = (cx, cy)

        # Build transformed selected segment
        moving = distal_piece if segment_choice == "distal" else proximal_piece
        fixed = proximal_piece if segment_choice == "distal" else distal_piece

        moved = apply_affine(moving, dx=dx, dy=dy, rot_deg=rotate_deg, center=center)

        # Composite back: fixed first, then moved
        composed = Image.new("RGBA", base_img.size, (0,0,0,0))
        composed = Image.alpha_composite(composed, fixed)
        composed = Image.alpha_composite(composed, moved)

        # Build a downscaled preview
        transformed_preview = composed.resize((canvas_width, canvas_height), Image.BICUBIC)

        st.image(transformed_preview, caption=f"Transformed ({segment_choice} moved)", use_container_width=True)

        # Export buttons
        params = asdict(st.session_state.params)
        params["polygon_points"] = poly_pts
        if st.session_state.px_per_mm:
            params["px_per_mm"] = st.session_state.px_per_mm

        # CSV export of params
        df_params = pd.DataFrame([params])
        csv_bytes = df_params.to_csv(index=False).encode("utf-8")
        st.download_button("Download parameters CSV", data=csv_bytes, file_name="osteotomy_params.csv", mime="text/csv")

        # Image export
        buf = io.BytesIO()
        composed.save(buf, format="PNG")
        st.download_button("Download transformed image (PNG)", data=buf.getvalue(), file_name="osteotomy_transformed.png", mime="image/png")
    else:
        st.caption("Tip: Use 'Osteotomy (lasso polygon)' to cut and move a segment.")

# Footer
st.markdown("---")
st.caption("Streamlit Bone Ninja — simplified demo for osteotomy planning | Cyan outlines during drawing for visibility | Use Ruler or Angle for measurements.")

