# streamlit_app.py
import io, math
from typing import List, Tuple, Optional
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – snappy single image", layout="wide")

Pt = Tuple[float, float]
Line = List[Pt]  # always [p0, p1]

# ---------------- helpers ----------------
def load_rgba(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(img).convert("RGBA")

def angle_deg(p0: Pt, p1: Pt) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    a = math.degrees(math.atan2(-dy, dx))  # y-down
    if a < 0: a += 360.0
    return a

def polygon_mask(size: Tuple[int,int], pts: List[Pt]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def centroid(pts: List[Pt]) -> Optional[Pt]:
    if len(pts) < 3: return None
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    a=cx=cy=0.0
    for i in range(len(pts)):
        j = (i+1) % len(pts)
        cross = x[i]*y[j] - x[j]*y[i]
        a += cross; cx += (x[i]+x[j])*cross; cy += (y[i]+y[j])*cross
    a *= 0.5
    if abs(a) < 1e-9: return None
    return (cx/(6*a), cy/(6*a))

def apply_affine_fragment(moving: Image.Image,
                          dx: float, dy: float,
                          rot_deg: float, center_xy: Pt) -> Image.Image:
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", moving.size, (0,0,0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def rotate_point(p: Tuple[float, float], center: Tuple[float, float], deg: float) -> Tuple[float, float]:
    """Rotate point p around center by +deg using SCREEN (y-down) coordinates.

    Matches Pillow's rotate(+deg), which is visually CCW on screen.
    """
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    cx, cy = center
    x, y = p[0] - cx, p[1] - cy
    # y-down rotation matrix (same as used for the image fragment)
    xr =  x * c + y * s
    yr = -x * s + y * c
    return (cx + xr, cy + yr)

def transform_line(line: List[Tuple[float, float]],
                   center: Tuple[float, float],
                   dx: float, dy: float, theta: float) -> List[Tuple[float, float]]:
    """Rotate line around center by +theta (y-down CCW), then translate by (dx, dy)."""
    p0 = rotate_point(line[0], center, theta)
    p1 = rotate_point(line[1], center, theta)
    return [(p0[0] + dx, p0[1] + dy), (p1[0] + dx, p1[1] + dy)]

def safe_width_slider(default_hint: int, uploaded_img: Optional[Image.Image]) -> int:
    """
    Build a width slider that never throws:
    - min_value = 200
    - max_value = min(1800, image_width) but at least 201
    - default clamped into [min,max]
    """
    min_w = 200
    if uploaded_img is None:
        max_w = 1200
    else:
        iw = uploaded_img.size[0]
        max_w = max(min_w + 1, min(1800, iw))

    default = max(min_w + 1, min(default_hint, max_w))
    return st.sidebar.slider("Preview width", min_value=min_w, max_value=max_w,
                             value=default, step=50)

# ---------------- state ----------------
ss = st.session_state
defaults = dict(
    dispw=1100,
    tool="Polygon",
    poly=[], poly_closed=False,
    hinge=None, cora=None,
    prox_axis=[], dist_axis=[],
    prox_joint=[], dist_joint=[],
    move_segment="distal",
    dx=0, dy=0, theta=0,
)
for k,v in defaults.items(): ss.setdefault(k, v)

# ---------------- sidebar ----------------
st.sidebar.header("Load image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool = st.sidebar.radio(
    "Tool",
    ["Polygon","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"],
    index=["Polygon","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"].index(ss.tool),
)

c1,c2,c3,c4 = st.sidebar.columns(4)
if c1.button("Reset poly"):   ss.poly=[]; ss.poly_closed=False
if c2.button("Reset axes"):   ss.prox_axis=[]; ss.dist_axis=[]
if c3.button("Reset joints"): ss.prox_joint=[]; ss.dist_joint=[]
if c4.button("Clear points"): ss.hinge=None; ss.cora=None

ss.move_segment = st.sidebar.radio(
    "Move which part after osteotomy?",
    ["distal","proximal"], horizontal=True,
    index=0 if ss.move_segment=="distal" else 1
)

# We need the image (or a stub) to size safely
probe_img = None
if up:   # load a tiny header only (cheap)
    probe_img = load_rgba(up.getvalue())

ss.dispw = safe_width_slider(ss.dispw, probe_img)

st.sidebar.markdown("---")
ss.dx    = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy    = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)

# ---------------- main ----------------
if not up:
    st.info("Upload an image to begin.")
    st.stop()

img = load_rgba(up.getvalue())
W,H = img.size
scale = min(ss.dispw/float(W), 1.0)
disp = img.resize((int(round(W*scale)), int(round(H*scale))), Image.NEAREST)

# --- build composite (apply osteotomy in display space) ---
composite = disp.copy()
center_for_motion: Pt = ss.hinge or centroid(ss.poly) or (disp.size[0]/2.0, disp.size[1]/2.0)

if ss.poly_closed and len(ss.poly) >= 3:
    m = polygon_mask(disp.size, ss.poly)
    inv = ImageOps.invert(m)
    prox = Image.new("RGBA", disp.size, (0,0,0,0)); prox.paste(disp, (0,0), inv)
    dist = Image.new("RGBA", disp.size, (0,0,0,0)); dist.paste(disp, (0,0), m)
    moving = dist if ss.move_segment=="distal" else prox
    fixed  = prox if ss.move_segment=="distal" else dist
    moved  = apply_affine_fragment(moving, ss.dx, ss.dy, ss.theta, center_for_motion)
    base   = Image.new("RGBA", disp.size, (0,0,0,0))
    base.alpha_composite(fixed)
    base.alpha_composite(moved)
    composite = base

# --- draw overlay (moving side axes/joints transformed in the same direction) ---
def overlay_img() -> Image.Image:
    img = composite.convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    # polygon (nodes + edges)
    if ss.poly:
        if len(ss.poly) >= 2: d.line(ss.poly, fill=(0,255,255,255), width=2)
        if ss.poly_closed and len(ss.poly) >= 3:
            d.line([ss.poly[-1], ss.poly[0]], fill=(0,255,255,255), width=2)
        for p in ss.poly:
            d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=(0,255,255,200))

    # decide what to draw (transform moving side for drawing only)
    prox_axis = ss.prox_axis[:]; dist_axis = ss.dist_axis[:]
    prox_joint = ss.prox_joint[:]; dist_joint = ss.dist_joint[:]

    if ss.poly_closed and len(ss.poly) >= 3:
        if ss.move_segment == "distal":
            if len(dist_axis)==2:  dist_axis  = transform_line(dist_axis,  center_for_motion, ss.dx, ss.dy, ss.theta)
            if len(dist_joint)==2: dist_joint = transform_line(dist_joint, center_for_motion, ss.dx, ss.dy, ss.theta)
        else:
            if len(prox_axis)==2:  prox_axis  = transform_line(prox_axis,  center_for_motion, ss.dx, ss.dy, ss.theta)
            if len(prox_joint)==2: prox_joint = transform_line(prox_joint, center_for_motion, ss.dx, ss.dy, ss.theta)


def _draw_line(line: Line, col):
    if len(line)==2:
        d.line(line, fill=col, width=3)
        for p in line:
            d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=col)

    _draw_line(prox_axis, (66,133,244,255))
    _draw_line(dist_axis, (221,0,221,255))
    _draw_line(prox_joint,(255,215,0,220))
    _draw_line(dist_joint,(255,215,0,220))

    if ss.hinge:
        x,y = ss.hinge
        d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)
    if ss.cora:
        x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)

    # angle readouts
    def _label(line: Line, y):
        if len(line)==2:
            a = angle_deg(line[0], line[1])
            d.rectangle([6,y-12,220,y+6], fill=(0,0,0,150))
            d.text((10,y-10), f"angle {a:.1f}°", fill=(255,255,255,230))
    y=8
    if len(prox_joint)==2: _label(prox_joint, y); y+=18
    if len(dist_joint)==2: _label(dist_joint, y); y+=18
    if len(prox_axis)==2:  _label(prox_axis,  y); y+=18
    if len(dist_axis)==2:  _label(dist_axis,  y); y+=18

    return img.convert("RGB")  # ensure RGB (avoids JPEG alpha errors)

overlay_rgb = overlay_img()
click = streamlit_image_coordinates(overlay_rgb, width=overlay_rgb.width, key="click")

# --- click handling (instant: update state and rerun) ---
if click and "x" in click and "y" in click:
    px, py = float(click["x"]), float(click["y"])
    p = (px, py)

    if ss.tool == "Polygon":
        if not ss.poly_closed:
            if len(ss.poly) >= 3:
                x0,y0 = ss.poly[0]
                if (px-x0)**2 + (py-y0)**2 <= 10**2:
                    ss.poly_closed = True
                else:
                    ss.poly.append(p)
            else:
                ss.poly.append(p)

    elif ss.tool == "Prox axis":
        if len(ss.prox_axis) < 1: ss.prox_axis = [p]
        elif len(ss.prox_axis) == 1: ss.prox_axis.append(p)
        else: ss.prox_axis = [p]

    elif ss.tool == "Dist axis":
        if len(ss.dist_axis) < 1: ss.dist_axis = [p]
        elif len(ss.dist_axis) == 1: ss.dist_axis.append(p)
        else: ss.dist_axis = [p]

    elif ss.tool == "Prox joint":
        if len(ss.prox_joint) < 1: ss.prox_joint = [p]
        elif len(ss.prox_joint) == 1: ss.prox_joint.append(p)
        else: ss.prox_joint = [p]

    elif ss.tool == "Dist joint":
        if len(ss.dist_joint) < 1: ss.dist_joint = [p]
        elif len(ss.dist_joint) == 1: ss.dist_joint.append(p)
        else: ss.dist_joint = [p]

    elif ss.tool == "HINGE":
        ss.hinge = p
    elif ss.tool == "CORA":
        ss.cora = p

    # immediate refresh (no second click needed)
    st.rerun()

with st.expander("Status / help", expanded=False):
    st.write(f"**Tool**: {ss.tool}  |  Polygon closed: {ss.poly_closed}")
    st.write("Close polygon by clicking within ~10 px of the first node. "
             "Hinge is the rotation center; the selected movement segment’s axes/joints follow the fragment.")
