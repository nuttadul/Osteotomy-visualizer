import io, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
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
       By convention we call the polygon area the 'distal' piece."""
    size = img_rgba.size
    mask = polygon_mask(size, poly)
    inv  = ImageOps.invert(mask)
    prox = Image.new("RGBA", size, (0,0,0,0)); prox.paste(img_rgba, (0,0), inv)
    dist = Image.new("RGBA", size, (0,0,0,0)); dist.paste(img_rgba, (0,0), mask)
    return prox, dist

# ----------------------------- State -----------------------------
@dataclass
class Axes:
    joint_tangent: Optional[Line] = None   # "joint line" (tangent)
    axis_line: Optional[Line] = None       # anatomical/mechanical axis

@dataclass
class AppState:
    poly: List[Pt] = field(default_factory=list)
    poly_closed: bool = False
    hinge: Optional[Pt] = None

    prox_axes: Axes = field(default_factory=Axes)
    dist_axes: Axes = field(default_factory=Axes)

    # transform
    move_segment: str = "distal"   # "distal" or "proximal"
    dx: float = 0.0
    dy: float = 0.0
    theta: float = 0.0             # degrees (CCW)

    # display scaling
    disp_w: int = 1100

def ss() -> AppState:
    if "S" not in st.session_state:
        st.session_state.S = AppState()
    return st.session_state.S  # type: ignore

# ----------------------------- Drawer -----------------------------
def draw_overlay(base_rgb: Image.Image, S: AppState) -> Image.Image:
    """Draw polygon, hinge, axes and simulated movement result onto a copy."""
    out = base_rgb.convert("RGBA")
    W,H = out.size

    # 1) draw persisted graphics on top of original (for orientation)
    draw0 = ImageDraw.Draw(out, "RGBA")

    # polygon vertices
    if S.poly:
        for p in S.poly:
            r = 4; x,y = p
            draw0.ellipse([x-r,y-r,x+r,y+r], fill=(0,255,255,220))
        # polygon outline
        if len(S.poly) >= 2:
            draw0.line(S.poly, fill=(0,255,255,180), width=2)
            if S.poly_closed and len(S.poly) >= 3:
                draw0.line([S.poly[-1], S.poly[0]], fill=(0,255,255,180), width=2)

    # 2) simulate osteotomy if polygon closed + hinge is defined
    composed = Image.new("RGBA", (W,H), (0,0,0,0))
    base_rgba = base_rgb.convert("RGBA")
    if S.poly_closed and S.hinge is not None and len(S.poly) >= 3:
        prox, dist = split_by_polygon(base_rgba, S.poly)
        moving  = dist if S.move_segment == "distal" else prox
        fixed   = prox if S.move_segment == "distal" else dist

        # apply transform to moving
        moved = Image.new("RGBA", (W,H), (0,0,0,0))
        rot   = moving.rotate(S.theta, resample=Image.BICUBIC, center=S.hinge, expand=False)
        moved.alpha_composite(rot, (int(round(S.dx)), int(round(S.dy))))

        composed.alpha_composite(fixed)
        composed.alpha_composite(moved)
    else:
        composed.alpha_composite(base_rgba)

    # 3) redraw axes on top, transforming the one that belongs to the moving part
    draw = ImageDraw.Draw(composed, "RGBA")
    green = (90, 200, 90, 255)
    blue  = (70, 140, 255, 255)
    mag   = (200, 70, 200, 255)
    orange= (255, 170, 0, 255)

    def put_line(line: Optional[Line], col):
        if not line: return
        draw.line([to_int(line[0]), to_int(line[1])], fill=col, width=3)

    # joint tangent & axes — transform the set that moves
    if S.move_segment == "distal":
        # distal transforms, proximal stays
        if S.dist_axes.joint_tangent:
            jt = apply_affine_to_line(S.dist_axes.joint_tangent, S.hinge or (0,0), S.dx,S.dy,S.theta)
            put_line(jt, orange)
        if S.dist_axes.axis_line:
            ax = apply_affine_to_line(S.dist_axes.axis_line, S.hinge or (0,0), S.dx,S.dy,S.theta)
            put_line(ax, mag)
        put_line(S.prox_axes.joint_tangent, orange)
        put_line(S.prox_axes.axis_line, blue)
    else:
        # proximal transforms, distal stays
        if S.prox_axes.joint_tangent:
            jt = apply_affine_to_line(S.prox_axes.joint_tangent, S.hinge or (0,0), S.dx,S.dy,S.theta)
            put_line(jt, orange)
        if S.prox_axes.axis_line:
            ax = apply_affine_to_line(S.prox_axes.axis_line, S.hinge or (0,0), S.dx,S.dy,S.theta)
            put_line(ax, blue)
        put_line(S.dist_axes.joint_tangent, orange)
        put_line(S.dist_axes.axis_line, mag)

    # draw hinge
    if S.hinge:
        x,y = S.hinge
        draw.ellipse([x-6,y-6,x+6,y+6], outline=(255,165,0,255), width=3)
        draw.line([(x-12,y),(x+12,y)], fill=(255,165,0,200), width=1)
        draw.line([(x,y-12),(x,y+12)], fill=(255,165,0,200), width=1)

    # angle labels for axes & joint tangents (if present)
    def put_angle(label: str, line: Optional[Line], xy: Optional[Pt]=None, col=(255,255,255,230)):
        if not line: return
        ang = line_angle(line)
        p = xy or line[1]
        draw.rectangle([p[0]+6, p[1]-18, p[0]+110, p[1]+4], fill=(0,0,0,140))
        draw.text((p[0]+8, p[1]-16), f"{label}: {ang:.1f}°", fill=col)

    if S.prox_axes.joint_tangent: put_angle("prox joint", S.prox_axes.joint_tangent)
    if S.prox_axes.axis_line:     put_angle("prox axis",  S.prox_axes.axis_line, col=blue)
    if S.dist_axes.joint_tangent: put_angle("dist joint", S.dist_axes.joint_tangent)
    if S.dist_axes.axis_line:     put_angle("dist axis",  S.dist_axes.axis_line, col=mag)

    return composed.convert("RGB")   # ensure RGB for streamlit_image_coordinates

# ----------------------------- Tools -----------------------------
TOOLS = [
    "Polygon",
    "Hinge",
    "Prox joint line",
    "Prox axis",
    "Dist joint line",
    "Dist axis",
]

def add_point_to_polygon(S: AppState, click: Pt, close_eps: float=10.0):
    """Immediate, one-click polygon node add. Closes loop if near first."""
    if not S.poly_closed:
        S.poly.append(click)
        if len(S.poly) >= 3 and distance(click, S.poly[0]) <= close_eps:
            S.poly[-1] = S.poly[0]  # snap
            S.poly_closed = True

def add_line_endpoint(current: Optional[Line], click: Pt) -> Optional[Line]:
    """Create/complete a 2-point line in one or two clicks. Immediate feedback."""
    if current is None:
        return (click, click)  # start; will be replaced by second click
    (p0,p1) = current
    if p0 == p1:
        return (p0, click)     # second click sets the end
    # If already completed, start a new line at this click
    return (click, click)

# ----------------------------- App -----------------------------
st.set_page_config(page_title="Osteotomy Visualizer (single-image, snappy)", layout="wide")
st.title("Osteotomy Visualizer (single image, snappy click)")

S = ss()

# Sidebar
st.sidebar.subheader("1) Load image")
up = st.sidebar.file_uploader("X-ray image", type=["png","jpg","jpeg","tif","tiff"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

# Decode, orient, set display size
img_rgba = Image.open(io.BytesIO(up.getvalue()))
img_rgba = ImageOps.exif_transpose(img_rgba).convert("RGBA")
W,H = img_rgba.size
S.disp_w = st.sidebar.slider("Display width", 600, min(1800, W), min(1100, W), 50)
scale   = S.disp_w / float(W)
disp_h  = int(round(H * scale))
base_rgb = img_rgba.resize((S.disp_w, disp_h), Image.NEAREST).convert("RGB")

# Tool selection
st.sidebar.subheader("2) Draw tools")
tool = st.sidebar.radio("Active tool", TOOLS, index=0, horizontal=False)

# Movement controls
st.sidebar.subheader("3) Segment transform")
S.move_segment = st.sidebar.radio("Which part moves?", ["distal","proximal"], index=0, horizontal=True)
S.dx   = st.sidebar.slider("ΔX (px)", -1000, 1000, int(S.dx), 1)
S.dy   = st.sidebar.slider("ΔY (px)", -1000, 1000, int(S.dy), 1)
S.theta= st.sidebar.slider("Rotate (°)", -180, 180, int(S.theta), 1)

# Reset controls
colA,colB,colC,colD = st.sidebar.columns(4)
if colA.button("Undo poly"):
    if S.poly: S.poly.pop()
    if len(S.poly) < 3: S.poly_closed = False
if colB.button("Clear poly"):
    S.poly.clear(); S.poly_closed=False
if colC.button("Clear lines"):
    S.prox_axes = Axes(); S.dist_axes = Axes()
if colD.button("Clear hinge"):
    S.hinge = None

# Prepare a **single** composite image (RGB) with all current drawings
composite = draw_overlay(base_rgb, S)

# Capture click on that same single image (IMMEDIATE: Streamlit reruns on click)
st.subheader("Working image (click to add / set with current tool)")
click = streamlit_image_coordinates(composite, width=composite.width, key="click-main")

def from_disp_to_orig(p: Pt) -> Pt:
    """Convert a click on the displayed (resized) image into original pixel coords,
       but since we draw & simulate entirely in display coords, we stay in display coords.
       i.e., we treat display coordinates as the working space everywhere."""
    return p

# Apply click immediately to the active tool
if click and "x" in click and "y" in click:
    pdisp = (float(click["x"]), float(click["y"]))

    if tool == "Polygon":
        add_point_to_polygon(S, pdisp)

    elif tool == "Hinge":
        S.hinge = pdisp

    elif tool == "Prox joint line":
        S.prox_axes.joint_tangent = add_line_endpoint(S.prox_axes.joint_tangent, pdisp)

    elif tool == "Prox axis":
        S.prox_axes.axis_line = add_line_endpoint(S.prox_axes.axis_line, pdisp)

    elif tool == "Dist joint line":
        S.dist_axes.joint_tangent = add_line_endpoint(S.dist_axes.joint_tangent, pdisp)

    elif tool == "Dist axis":
        S.dist_axes.axis_line = add_line_endpoint(S.dist_axes.axis_line, pdisp)

    # Nothing else to do — Streamlit already reruns, so result is shown instantly.

# Helpful instructions
st.caption(
    """
**Tips**
- Select a tool, then click on the image to add **nodes/points immediately**.
- **Polygon**: click around the osteotomy; when your last click is near the first point it **auto-closes**.
- **Hinge**: click to set the rotation center (crosshair).
- **Prox/Dist joint line**: two clicks make a line; a 3rd click starts a new one.
- **Prox/Dist axis**: same as lines. The axis belonging to the **moving** segment rotates & translates with it.
- Use the sliders (ΔX, ΔY, Rotate) to simulate the osteotomy.
"""
)

# Download buttons (composite only, in display pixels)
buf = io.BytesIO()
draw_overlay(base_rgb, S).save(buf, format="PNG")
st.download_button("Download current view (PNG)", data=buf.getvalue(),
                   file_name="osteotomy_view.png", mime="image/png")
