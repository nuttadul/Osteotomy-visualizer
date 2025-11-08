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
    # Pillow rotate is CCW in screen (y-down) coordinates
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", moving.size, (0,0,0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def rotate_point(p: Pt, center: Pt, deg: float) -> Pt:
    """Rotate point p around center by +deg using SCREEN (y-down) coordinates."""
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    cx, cy = center
    x, y = p[0] - cx, p[1] - cy
    xr =  x * c + y * s
    yr = -x * s + y * c
    return (cx + xr, cy + yr)

def transform_line(line: Line, center: Pt, dx: float, dy: float, theta: float) -> Line:
    p0 = rotate_point(line[0], center, theta)
    p1 = rotate_point(line[1], center, theta)
    return [(p0[0] + dx, p0[1] + dy), (p1[0] + dx, p1[1] + dy)]

def safe_width_slider(default_hint: int, uploaded_img: Optional[Image.Image]) -> int:
    min_w = 200
    if uploaded_img is None:
        max_w = 1200
    else:
        iw = uploaded_img.size[0]
        max_w = max(min_w + 1, min(1800, iw))
    default = max(min_w + 1, min(default_hint, max_w))
    return st.sidebar.slider("Preview width", min_value=min_w, max_value=max_w,
                             value=default, step=50)

# ----- vector / geometry helpers for angles & intersections -----
def _vec(p0: Pt, p1: Pt) -> Pt:
    return (p1[0]-p0[0], p1[1]-p0[1])

def _norm(v: Pt) -> float:
    return (v[0]**2 + v[1]**2) ** 0.5

def _unit(v: Pt) -> Pt:
    n = _norm(v)
    if n == 0: return (0.0, 0.0)
    return (v[0]/n, v[1]/n)

def _perp(v: Pt) -> Pt:
    # 90° CCW in screen coords
    return (-v[1], v[0])

def angle_between_lines(l1: Line, l2: Line) -> Optional[Tuple[float,float]]:
    """Return (small, large) angles in degrees between two infinite lines (0..180)."""
    if len(l1)!=2 or len(l2)!=2: return None
    v1 = _unit(_vec(l1[0], l1[1]))
    v2 = _unit(_vec(l2[0], l2[1]))
    if _norm(v1)==0 or _norm(v2)==0: return None
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    dot = max(-1.0, min(1.0, dot))
    small = math.degrees(math.acos(dot))  # 0..180
    large = 180.0 - small
    return (small, large)

def line_intersection(l1: Line, l2: Line) -> Optional[Pt]:
    """Intersection of infinite lines; returns None if parallel/degenerate."""
    if len(l1)!=2 or len(l2)!=2: return None
    x1,y1 = l1[0]; x2,y2 = l1[1]
    x3,y3 = l2[0]; x4,y4 = l2[1]
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9: return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
    return (px, py)

def bubble(d: ImageDraw.ImageDraw, anchor: Pt, text: str):
    """Small dark label near a point."""
    pad = 4
    # crude text width estimate; avoids font dependency
    tw = max(40, int(8 * max(1, len(text)) * 0.55))
    th = 16
    bx0, by0 = anchor[0] + 8, anchor[1] - 8
    bx1, by1 = bx0 + tw + pad*2, by0 + th
    d.rectangle([bx0,by0,bx1,by1], fill=(0,0,0,170))
    d.text((bx0+pad, by0+2), text, fill=(255,255,255,230))

# ----- map sliders (ΔX ⟂ prox axis, ΔY ∥ prox axis) to screen dx,dy -----
def map_sliders_to_screen(dx_slider: float, dy_slider: float, prox_axis: Line) -> Tuple[float,float]:
    """ΔY moves parallel to proximal axis, ΔX moves perpendicular (CCW 90°)."""
    if len(prox_axis) == 2:
        v = _unit(_vec(prox_axis[0], prox_axis[1]))      # parallel
        n = _unit(_perp(v))                               # perpendicular (CCW)
        dx = dx_slider * n[0] + dy_slider * v[0]
        dy = dx_slider * n[1] + dy_slider * v[1]
        return (dx, dy)
    # fallback to screen axes if prox axis missing
    return (dx_slider, dy_slider)

# ---------------- state ----------------
ss = st.session_state
defaults = dict(
    dispw=1100,
    tool="Osteotomy",
    poly=[], poly_closed=False,
    hinge=None, cora=None,
    prox_axis=[], dist_axis=[],
    prox_joint=[], dist_joint=[],
    move_segment="distal",
    dx=0, dy=0, theta=0.0,
)
for k,v in defaults.items(): ss.setdefault(k, v)

# ---------------- sidebar ----------------
st.sidebar.title("Osteotomy visualizer")  # << title requested

st.sidebar.header("Load image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool = st.sidebar.radio(
    "Tool",
    ["Osteotomy","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"],
    index=["Osteotomy","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"].index(ss.tool),
)

st.sidebar.markdown("**Delete a single item**")
del_choice = st.sidebar.selectbox(
    "Choose one to clear",
    ["(none)","Prox axis","Dist axis","Prox joint","Dist joint"]
)
if st.sidebar.button("Delete selected"):
    if del_choice == "Prox axis":  ss.prox_axis = []
    if del_choice == "Dist axis":  ss.dist_axis = []
    if del_choice == "Prox joint": ss.prox_joint = []
    if del_choice == "Dist joint": ss.dist_joint = []

st.sidebar.markdown("---")
c1,c2,c3,c4 = st.sidebar.columns(4)
if c1.button("Reset osteotomy"): ss.poly=[]; ss.poly_closed=False
if c2.button("Reset axes"):       ss.prox_axis=[]; ss.dist_axis=[]
if c3.button("Reset joints"):     ss.prox_joint=[]; ss.dist_joint=[]
if c4.button("Clear points"):     ss.hinge=None; ss.cora=None

ss.move_segment = st.sidebar.radio(
    "Move which part after osteotomy?",
    ["distal","proximal"], horizontal=True,
    index=0 if ss.move_segment=="distal" else 1
)

probe_img = load_rgba(up.getvalue()) if up else None
ss.dispw = safe_width_slider(ss.dispw, probe_img)

st.sidebar.markdown("---")
ss.dx    = st.sidebar.slider("ΔX (⟂ prox axis)  px", -500, 500, ss.dx, 1)
ss.dy    = st.sidebar.slider("ΔY (∥ prox axis) px", -500, 500, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -60.0, 60.0, float(ss.theta), 0.2)

# ---------------- main ----------------
if not up:
    st.info("Upload an image to begin.")
    st.stop()

img = load_rgba(up.getvalue())
W,H = img.size
scale = min(ss.dispw/float(W), 1.0)
disp = img.resize((int(round(W*scale)), int(round(H*scale))), Image.NEAREST)

# Map sliders to screen displacement using proximal axis
dx_screen, dy_screen = map_sliders_to_screen(ss.dx, ss.dy, ss.prox_axis)

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
    moved  = apply_affine_fragment(moving, dx_screen, dy_screen, ss.theta, center_for_motion)
    base   = Image.new("RGBA", disp.size, (0,0,0,0))
    base.alpha_composite(fixed)
    base.alpha_composite(moved)
    composite = base

# --- draw overlay (moving side axes/joints transformed in the same direction) ---
def overlay_img() -> Image.Image:
    img = composite.convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    # osteotomy polygon (nodes + edges)
    if ss.poly:
        if len(ss.poly) >= 2:
            d.line(ss.poly, fill=(0,255,255,255), width=2)
        if ss.poly_closed and len(ss.poly) >= 3:
            d.line([ss.poly[-1], ss.poly[0]], fill=(0,255,255,255), width=2)
        for p in ss.poly:
            d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=(0,255,255,200))

    # decide what to draw (transform moving side for drawing only)
    prox_axis = ss.prox_axis[:]; dist_axis = ss.dist_axis[:]
    prox_joint = ss.prox_joint[:]; dist_joint = ss.dist_joint[:]

    if ss.poly_closed and len(ss.poly) >= 3:
        if ss.move_segment == "distal":
            if len(dist_axis)==2:  dist_axis  = transform_line(dist_axis,  center_for_motion, dx_screen, dy_screen, ss.theta)
            if len(dist_joint)==2: dist_joint = transform_line(dist_joint, center_for_motion, dx_screen, dy_screen, ss.theta)
        else:
            if len(prox_axis)==2:  prox_axis  = transform_line(prox_axis,  center_for_motion, dx_screen, dy_screen, ss.theta)
            if len(prox_joint)==2: prox_joint = transform_line(prox_joint, center_for_motion, dx_screen, dy_screen, ss.theta)

    def _draw_line(line: Line, col, label: str):
        if len(line) >= 1:
            p0 = line[0]
            d.ellipse([p0[0]-4, p0[1]-4, p0[0]+4, p0[1]+4], fill=col)
        if len(line) == 2:
            d.line(line, fill=col, width=3)
            for p in line:
                d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=col)
            # line name at midpoint
            mid = ((line[0][0]+line[1][0])/2.0, (line[0][1]+line[1][1])/2.0)
            bubble(d, mid, label)

    _draw_line(prox_axis, (66,133,244,255), "prox axis")
    _draw_line(dist_axis, (221,0,221,255), "dist axis")
    _draw_line(prox_joint,(255,215,0,220), "prox joint")
    _draw_line(dist_joint,(255,215,0,220), "dist joint")

    # angle reports (both small & large) at intersections
    def _angle_pair(name: str, l1: Line, l2: Line, dy_offset: int = 0):
        ab = angle_between_lines(l1, l2)
        if not ab: return
        inter = line_intersection(l1, l2)
        if not inter:
            # fallback: midpoint between the two midpoints
            if len(l1)==2 and len(l2)==2:
                m1 = ((l1[0][0]+l1[1][0])/2.0, (l1[0][1]+l1[1][1])/2.0)
                m2 = ((l2[0][0]+l2[1][0])/2.0, (l2[0][1]+l2[1][1])/2.0)
                inter = ((m1[0]+m2[0])/2.0, (m1[1]+m2[1])/2.0)
            else:
                return
        # small & large
        small, large = ab
        bubble(d, (inter[0], inter[1] + dy_offset), f"{name}  small:{small:.1f}°  large:{large:.1f}°")

    # 1) prox joint vs prox axis
    if len(prox_axis)==2 and len(prox_joint)==2:
        _angle_pair("prox joint↔axis", prox_joint, prox_axis, dy_offset=-18)
    # 2) dist joint vs dist axis
    if len(dist_axis)==2 and len(dist_joint)==2:
        _angle_pair("dist joint↔axis", dist_joint, dist_axis, dy_offset=-18)
    # 3) prox axis vs dist axis
    if len(prox_axis)==2 and len(dist_axis)==2:
        _angle_pair("prox axis↔dist axis", prox_axis, dist_axis, dy_offset=12)

    if ss.hinge:
        x,y = ss.hinge
        d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)
    if ss.cora:
        x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)

    return img.convert("RGB")  # JPEG-friendly (no alpha)

overlay_rgb = overlay_img()
click = streamlit_image_coordinates(overlay_rgb, width=overlay_rgb.size[0], key="click")

# --- click handling (instant: update state and rerun) ---
if click and "x" in click and "y" in click:
    px, py = float(click["x"]), float(click["y"])
    p = (px, py)

    if ss.tool == "Osteotomy":
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

    st.rerun()

with st.expander("Status / help", expanded=False):
    st.write(f"**Tool**: {ss.tool}  |  Osteotomy closed: {ss.poly_closed}")
    st.write("Click once to place a node; twice to complete a segment. "
             "Close the osteotomy by clicking near the first node. "
             "ΔY slides parallel to the proximal axis; ΔX slides perpendicular to it. "
             "Angle bubbles show small & large angles for joint↔axis and axis↔axis.")
