import io, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------- Utilities -----------------------------
Pt = Tuple[float, float]
Line = Tuple[Pt, Pt]

def to_int(p: Pt) -> Tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))

def distance(a: Pt, b: Pt) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def line_angle(line: Line) -> float:
    (x1,y1),(x2,y2) = line
    return math.degrees(math.atan2(y2-y1, x2-x1))

def rotate_point(p: Pt, center: Pt, deg: float) -> Pt:
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    x, y = p[0]-center[0], p[1]-center[1]
    return (center[0] + x*c - y*s, center[1] + x*s + y*c)

def translate(p: Pt, dx: float, dy: float) -> Pt:
    return (p[0]+dx, p[1]+dy)

def apply_affine_to_line(line: Line, center: Pt, dx: float, dy: float, deg: float) -> Line:
    return (
        translate(rotate_point(line[0], center, deg), dx, dy),
        translate(rotate_point(line[1], center, deg), dx, dy),
    )

def polygon_mask(size: Tuple[int,int], pts: List[Pt]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def split_by_polygon(img_rgba: Image.Image, poly: List[Pt]) -> Tuple[Image.Image, Image.Image]:
    """Return (prox_segment, dist_segment) by polygon selection.
       By convention the polygon area is the 'distal' piece."""
    size = img_rgba.size
    mask = polygon_mask(size, poly)
    inv  = ImageOps.invert(mask)
    prox = Image.new("RGBA", size, (0,0,0,0)); prox.paste(img_rgba, (0,0), inv)
    dist = Image.new("RGBA", size, (0,0,0,0)); dist.paste(img_rgba, (0,0), mask)
    return prox, dist

# ----------------------------- State -----------------------------
@dataclass
class Axes:
    joint_tangent: Optional[Line] = None
    axis_line: Optional[Line] = None

@dataclass
class AppState:
    poly: List[Pt] = field(default_factory=list)
    poly_closed: bool = False
    hinge: Optional[Pt] = None

    prox_axes: Axes = field(default_factory=Axes)
    dist_axes: Axes = field(default_factory=Axes)

    move_segment: str = "distal"   # "distal" or "proximal"
    dx: float = 0.0
    dy: float = 0.0
    theta: float = 0.0             # degrees (CCW)

    disp_w: int = 1100             # display width (pixels)

def S() -> AppState:
    if "APPSTATE" not in st.session_state:
        st.session_state.APPSTATE = AppState()
    return st.session_state.APPSTATE  # type: ignore

# ----------------------------- Drawer -----------------------------
def draw_overlay(base_rgb: Image.Image, state: AppState) -> Image.Image:
    out = base_rgb.convert("RGBA")
    W,H = out.size
    draw0 = ImageDraw.Draw(out, "RGBA")

    # Polygon nodes/outline
    if state.poly:
        for p in state.poly:
            r = 4; x,y = p
            draw0.ellipse([x-r,y-r,x+r,y+r], fill=(0,255,255,220))
        if len(state.poly) >= 2:
            draw0.line(state.poly, fill=(0,255,255,180), width=2)
            if state.poly_closed and len(state.poly) >= 3:
                draw0.line([state.poly[-1], state.poly[0]], fill=(0,255,255,180), width=2)

    # Compose moved segment if polygon & hinge present
    composed = Image.new("RGBA", (W,H), (0,0,0,0))
    base_rgba = base_rgb.convert("RGBA")
    if state.poly_closed and state.hinge is not None and len(state.poly) >= 3:
        prox, dist = split_by_polygon(base_rgba, state.poly)
        moving  = dist if state.move_segment == "distal" else prox
        fixed   = prox if state.move_segment == "distal" else dist

        moved = Image.new("RGBA", (W,H), (0,0,0,0))
        rot   = moving.rotate(state.theta, resample=Image.BICUBIC, center=state.hinge, expand=False)
        moved.alpha_composite(rot, (int(round(state.dx)), int(round(state.dy))))

        composed.alpha_composite(fixed)
        composed.alpha_composite(moved)
    else:
        composed.alpha_composite(base_rgba)

    draw = ImageDraw.Draw(composed, "RGBA")
    green = (90, 200, 90, 255)
    blue  = (70, 140, 255, 255)
    mag   = (200, 70, 200, 255)
    orange= (255, 170, 0, 255)

    def put_line(line: Optional[Line], col):
        if not line: return
        draw.line([to_int(line[0]), to_int(line[1])], fill=col, width=3)

    # Transform the axes that belong to the moving part
    if state.move_segment == "distal":
        if state.dist_axes.joint_tangent:
            jt = apply_affine_to_line(state.dist_axes.joint_tangent, state.hinge or (0,0), state.dx,state.dy,state.theta)
            put_line(jt, orange)
        if state.dist_axes.axis_line:
            ax = apply_affine_to_line(state.dist_axes.axis_line, state.hinge or (0,0), state.dx,state.dy,state.theta)
            put_line(ax, mag)
        put_line(state.prox_axes.joint_tangent, orange)
        put_line(state.prox_axes.axis_line, blue)
    else:
        if state.prox_axes.joint_tangent:
            jt = apply_affine_to_line(state.prox_axes.joint_tangent, state.hinge or (0,0), state.dx,state.dy,state.theta)
            put_line(jt, orange)
        if state.prox_axes.axis_line:
            ax = apply_affine_to_line(state.prox_axes.axis_line, state.hinge or (0,0), state.dx,state.dy,state.theta)
            put_line(ax, blue)
        put_line(state.dist_axes.joint_tangent, orange)
        put_line(state.dist_axes.axis_line, mag)

    if state.hinge:
        x,y = state.hinge
        draw.ellipse([x-6,y-6,x+6,y+6], outline=(255,165,0,255), width=3)
        draw.line([(x-12,y),(x+12,y)], fill=(255,165,0,200), width=1)
        draw.line([(x,y-12),(x,y+12)], fill=(255,165,0,200), width=1)

    # Simple angle labels
    def put_angle(label: str, line: Optional[Line], xy: Optional[Pt]=None, col=(255,255,255,230)):
        if not line: return
        ang = line_angle(line)
        p = xy or line[1]
        draw.rectangle([p[0]+6, p[1]-18, p[0]+120, p[1]+4], fill=(0,0,0,140))
        draw.text((p[0]+8, p[1]-16), f"{label}: {ang:.1f}°", fill=col)

    if state.prox_axes.joint_tangent: put_angle("prox joint", state.prox_axes.joint_tangent)
    if state.prox_axes.axis_line:     put_angle("prox axis",  state.prox_axes.axis_line, col=blue)
    if state.dist_axes.joint_tangent: put_angle("dist joint", state.dist_axes.joint_tangent)
    if state.dist_axes.axis_line:     put_angle("dist axis",  state.dist_axes.axis_line, col=mag)

    return composed.convert("RGB")

# ----------------------------- Tools -----------------------------
TOOLS = [
    "Polygon",
    "Hinge",
    "Prox joint line",
    "Prox axis",
    "Dist joint line",
    "Dist axis",
]

def add_point_to_polygon(state: AppState, click: Pt, close_eps: float=10.0):
    if not state.poly_closed:
        state.poly.append(click)
        if len(state.poly) >= 3 and distance(click, state.poly[0]) <= close_eps:
            state.poly[-1] = state.poly[0]
            state.poly_closed = True

def add_line_endpoint(current: Optional[Line], click: Pt) -> Optional[Line]:
    if current is None:
        return (click, click)
    p0, p1 = current
    if p0 == p1:
        return (p0, click)
    return (click, click)

# ----------------------------- App -----------------------------
st.set_page_config(page_title="Osteotomy Visualizer (single-image, snappy)", layout="wide")
st.title("Osteotomy Visualizer (single image, snappy click)")

state = S()

# Sidebar: load
st.sidebar.subheader("1) Load image")
up = st.sidebar.file_uploader("X-ray image", type=["png","jpg","jpeg","tif","tiff"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

img_rgba = Image.open(io.BytesIO(up.getvalue()))
img_rgba = ImageOps.exif_transpose(img_rgba).convert("RGBA")
W,H = img_rgba.size

# ---- SAFE slider bounds (fix for your error)
max_w = max(1, min(W, 1800))                 # upper bound cannot be < 1
min_w = max(200, min(600, max_w))            # lower bound <= upper, at least 200
default_w = max(min_w, min(1100, max_w))     # inside [min_w, max_w]
step = max(1, min(50, max_w - min_w))        # at least 1, not larger than range
state.disp_w = st.sidebar.slider("Display width", min_w, max_w, default_w, step)

scale   = state.disp_w / float(W)
disp_h  = int(round(H * scale))
base_rgb = img_rgba.resize((state.disp_w, disp_h), Image.NEAREST).convert("RGB")

# Tools
st.sidebar.subheader("2) Draw tools")
tool = st.sidebar.radio("Active tool", TOOLS, index=0)

# Movement
st.sidebar.subheader("3) Segment transform")
state.move_segment = st.sidebar.radio("Which part moves?", ["distal","proximal"], index=0, horizontal=True)
state.dx    = st.sidebar.slider("ΔX (px)", -1000, 1000, int(state.dx), 1)
state.dy    = st.sidebar.slider("ΔY (px)", -1000, 1000, int(state.dy), 1)
state.theta = st.sidebar.slider("Rotate (°)", -180, 180, int(state.theta), 1)

# Reset
cA,cB,cC,cD = st.sidebar.columns(4)
if cA.button("Undo poly"):
    if state.poly: state.poly.pop()
    if len(state.poly) < 3: state.poly_closed = False
if cB.button("Clear poly"):
    state.poly.clear(); state.poly_closed=False
if cC.button("Clear lines"):
    state.prox_axes = Axes(); state.dist_axes = Axes()
if cD.button("Clear hinge"):
    state.hinge = None

# Working image
composite = draw_overlay(base_rgb, state)
st.subheader("Working image (click to add / set with current tool)")
click = streamlit_image_coordinates(composite, width=composite.width, key="click-main")

# Immediate handling
if click and "x" in click and "y" in click:
    p = (float(click["x"]), float(click["y"]))
    if tool == "Polygon":
        add_point_to_polygon(state, p)
    elif tool == "Hinge":
        state.hinge = p
    elif tool == "Prox joint line":
        state.prox_axes.joint_tangent = add_line_endpoint(state.prox_axes.joint_tangent, p)
    elif tool == "Prox axis":
        state.prox_axes.axis_line = add_line_endpoint(state.prox_axes.axis_line, p)
    elif tool == "Dist joint line":
        state.dist_axes.joint_tangent = add_line_endpoint(state.dist_axes.joint_tangent, p)
    elif tool == "Dist axis":
        state.dist_axes.axis_line = add_line_endpoint(state.dist_axes.axis_line, p)

st.caption(
    """
**Tips**
- Choose a tool, then click the image. Nodes/lines appear **immediately** (no second click).
- **Polygon** closes when your last click is close to the first vertex.
- **Hinge** sets the rotation center.
- Draw **joint lines** and **axes** with two clicks; a third click starts a new line.
- The axis of the **moving** segment translates & rotates with the osteotomy.
"""
)

# Download current view
buf = io.BytesIO()
draw_overlay(base_rgb, state).save(buf, format="PNG")
st.download_button("Download current view (PNG)", data=buf.getvalue(),
                   file_name="osteotomy_view.png", mime="image/png")
