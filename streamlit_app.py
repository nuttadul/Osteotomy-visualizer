# streamlit_app.py
import io, math
from typing import List, Tuple, Optional
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – snappy click tools (single image)", layout="wide")

# ------------------------------- helpers -------------------------------

Pt = Tuple[float, float]
Line = List[Pt]  # always [p0, p1] when present

def load_rgba(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(img).convert("RGBA")

def angle_deg(p0: Pt, p1: Pt) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    a = math.degrees(math.atan2(-(dy), dx))  # y-down screen
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

def apply_affine_fragment(moving: Image.Image,
                          dx: float, dy: float,
                          rot_deg: float, center_xy: Pt) -> Image.Image:
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", moving.size, (0,0,0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def rotate_point(p: Pt, center: Pt, deg: float) -> Pt:
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    x, y = p[0]-center[0], p[1]-center[1]
    return (center[0] + x*c - y*s, center[1] + x*s + y*c)

def transform_line(line: Line, center: Pt, dx: float, dy: float, theta: float) -> Line:
    return [
        (rotate_point(line[0], center, theta)[0] + dx, rotate_point(line[0], center, theta)[1] + dy),
        (rotate_point(line[1], center, theta)[0] + dx, rotate_point(line[1], center, theta)[1] + dy),
    ]

# --------------------------- session state -----------------------------

ss = st.session_state
defaults = dict(
    dispw=1100,
    tool="Polygon",
    last_click=None,   # debouncer
    render_nonce=0,    # forces component redraw without rerun

    poly=[],
    poly_closed=False,
    hinge=None,
    cora=None,
    prox_axis=[],   # [p0,p1]
    dist_axis=[],
    prox_joint=[],
    dist_joint=[],

    move_segment="distal",   # "distal" or "proximal"
    dx=0, dy=0, theta=0,
)
for k,v in defaults.items(): ss.setdefault(k,v)

# ------------------------------ sidebar --------------------------------

st.sidebar.header("Load image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
ss.tool = st.sidebar.radio("Tool", ["Polygon", "Prox axis", "Dist axis",
                                    "Prox joint", "Dist joint", "HINGE", "CORA"],
                           index=["Polygon","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"].index(ss.tool))

c1,c2,c3,c4 = st.sidebar.columns(4)
if c1.button("Reset poly"):    ss.poly=[]; ss.poly_closed=False
if c2.button("Reset axes"):    ss.prox_axis=[]; ss.dist_axis=[]
if c3.button("Reset joints"):  ss.prox_joint=[]; ss.dist_joint=[]
if c4.button("Clear points"):  ss.hinge=None; ss.cora=None

ss.move_segment = st.sidebar.radio("Move which part after osteotomy?",
                                   ["distal","proximal"], horizontal=True,
                                   index=0 if ss.move_segment=="distal" else 1)

# safe slider bounds (prevents width slider crashes on small images)
if up:
    tmp = load_rgba(up.getvalue())
    W0 = tmp.size[0]
    max_w = max(1, min(1800, W0))
    min_w = max(200, min(600, max_w))
    default_w = max(min_w, min(ss.dispw, max_w))
else:
    min_w, max_w, default_w = 600, 1800, ss.dispw

ss.dispw = st.sidebar.slider("Preview width", min_w, max_w, default_w, 50)

st.sidebar.markdown("---")
ss.dx    = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy    = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)

# ------------------------------ main -----------------------------------

if not up:
    st.info("Upload an image to begin.")
    st.stop()

img = load_rgba(up.getvalue())
W,H = img.size
scale = min(ss.dispw/float(W), 1.0)
disp = img.resize((int(round(W*scale)), int(round(H*scale))), Image.NEAREST)

# ---------- build composite (polygon transform) ----------
composite_for_display = disp.copy()
center_for_motion: Pt = ss.hinge or centroid(ss.poly) or (disp.size[0]/2.0, disp.size[1]/2.0)

if ss.poly_closed and len(ss.poly) >= 3:
    m_disp = polygon_mask(disp.size, ss.poly)
    inv_disp = ImageOps.invert(m_disp)
    prox_disp = Image.new("RGBA", disp.size, (0,0,0,0)); prox_disp.paste(disp, (0,0), inv_disp)
    dist_disp = Image.new("RGBA", disp.size, (0,0,0,0)); dist_disp.paste(disp, (0,0), m_disp)
    moving = dist_disp if ss.move_segment=="distal" else prox_disp
    fixed  = prox_disp if ss.move_segment=="distal" else dist_disp
    moved = apply_affine_fragment(moving, ss.dx, ss.dy, ss.theta, center_for_motion)
    base = Image.new("RGBA", disp.size, (0,0,0,0))
    base.alpha_composite(fixed)
    base.alpha_composite(moved)
    composite_for_display = base

# ---------- draw overlay (including transformed axes) ----------
def draw_overlay(base_img: Image.Image) -> Image.Image:
    show = base_img.convert("RGBA")
    d = ImageDraw.Draw(show, "RGBA")

    # polygon
    if ss.poly:
        if len(ss.poly) >= 2:
            d.line(ss.poly, fill=(0,255,255,255), width=2)
        if ss.poly_closed and len(ss.poly) >= 3:
            d.line([ss.poly[-1], ss.poly[0]], fill=(0,255,255,255), width=2)
        for p in ss.poly:
            d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=(0,255,255,200))

    def _draw_line(line: Line, col):
        if len(line) == 2:
            d.line(line, fill=col, width=3)
            for p in line:
                d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=col)

    # Axes / joints — transform the MOVING side only for drawing
    prox_axis = ss.prox_axis[:]
    dist_axis = ss.dist_axis[:]
    prox_joint = ss.prox_joint[:]
    dist_joint = ss.dist_joint[:]

    if ss.poly_closed and len(ss.poly) >= 3:
        if ss.move_segment == "distal":
            if len(dist_axis)==2:
                dist_axis = transform_line(dist_axis, center_for_motion, ss.dx, ss.dy, ss.theta)
            if len(dist_joint)==2:
                dist_joint = transform_line(dist_joint, center_for_motion, ss.dx, ss.dy, ss.theta)
        else:
            if len(prox_axis)==2:
                prox_axis = transform_line(prox_axis, center_for_motion, ss.dx, ss.dy, ss.theta)
            if len(prox_joint)==2:
                prox_joint = transform_line(prox_joint, center_for_motion, ss.dx, ss.dy, ss.theta)

    _draw_line(prox_axis, (66,133,244,255))
    _draw_line(dist_axis, (221,0,221,255))
    _draw_line(prox_joint, (255,215,0,220))
    _draw_line(dist_joint, (255,215,0,220))

    if ss.cora:
        x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)
    if ss.hinge:
        x,y=ss.hinge
        d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)

    # angle labels
    def _label_angle(line: Line, label_y):
        if len(line) == 2:
            a = angle_deg(line[0], line[1])
            d.rectangle([6,label_y-12,220,label_y+6], fill=(0,0,0,160))
            d.text((10,label_y-10), f"angle {a:.1f}°", fill=(255,255,255,230))

    ytick=8
    if len(prox_joint)==2: _label_angle(prox_joint, ytick); ytick+=18
    if len(dist_joint)==2: _label_angle(dist_joint, ytick); ytick+=18
    if len(prox_axis)==2:  _label_angle(prox_axis,  ytick); ytick+=18
    if len(dist_axis)==2:  _label_angle(dist_axis,  ytick); ytick+=18

    return show.convert("RGB")

# ---------- SNAPPY rendering: placeholder + re-render-on-click ----------
img_placeholder = st.empty()
overlay_img = draw_overlay(composite_for_display)

click = img_placeholder.image(streamlit_image_coordinates(
    overlay_img, width=overlay_img.width, key=f"click-{ss.render_nonce}"
))

# NOTE: streamlit_image_coordinates returns value directly; we need to call it outside st.image
click = streamlit_image_coordinates(overlay_img, width=overlay_img.width, key=f"click-{ss.render_nonce}")

if click and "x" in click and "y" in click:
    event = (click["x"], click["y"], overlay_img.width, overlay_img.height, ss.tool)
    if event != ss.last_click:
        ss.last_click = event
        px, py = float(click["x"]), float(click["y"])
        p = (px, py)

        # ---- tools (instant write to state) ----
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

        # re-render **immediately** with a new key (no rerun)
        ss.render_nonce += 1
        overlay_img = draw_overlay(composite_for_display)
        img_placeholder.empty()
        # draw a fresh clickable image (we don't consume its click result this pass)
        streamlit_image_coordinates(overlay_img, width=overlay_img.width, key=f"click-{ss.render_nonce}")

with st.expander("Status / help", expanded=False):
    st.write(f"**Tool**: {ss.tool}  |  Polygon closed: {ss.poly_closed}")
    st.write("Polygon closes when your click is within ~10 px of the first node. "
             "Hinge defines the rotation center. Moving segment’s axes follow the fragment.")
