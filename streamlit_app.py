# streamlit_app.py
import io, math
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – snappy click tools (single image)", layout="wide")

# ------------------------------- helpers -------------------------------

def load_rgba(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(img).convert("RGBA")

def angle_deg(p0: Tuple[float,float], p1: Tuple[float,float]) -> float:
    """Return absolute angle (deg) from p0->p1 (screen coords, y-down)."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    # y-down screen: atan2(-dy, dx) gives conventional mathematical angle
    a = math.degrees(math.atan2(-(dy), dx))
    if a < 0: a += 360.0
    return a

def polygon_mask(size: Tuple[int,int], pts: List[Tuple[float,float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def centroid(pts: List[Tuple[float,float]]) -> Optional[Tuple[float,float]]:
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
                          rot_deg: float, center_xy: Tuple[float,float]) -> Image.Image:
    """
    Rotate around center (screen y-down), then translate by dx,dy.
    Pillow rotate is CCW in y-down, which is what we want for on-screen.
    """
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", moving.size, (0,0,0,0))
    # translation is simply an offset for compositing
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def transform_points_y_down(points: List[Tuple[float,float]],
                            dx: float, dy: float,
                            rot_deg: float, center: Tuple[float,float]) -> List[Tuple[float,float]]:
    """Same rotation/translation used for the image fragment."""
    if not points: return []
    ang = math.radians(rot_deg)
    c, s = math.cos(ang), math.sin(ang)
    cx, cy = center
    out = []
    for (x, y) in points:
        x0, y0 = x - cx, y - cy
        xr =  x0*c + y0*s + cx + dx
        yr = -x0*s + y0*c + cy + dy
        out.append((xr, yr))
    return out

def draw_overlay(base: Image.Image, ss) -> Image.Image:
    """Return a copy of base + all shapes drawn."""
    show = base.copy()
    d = ImageDraw.Draw(show, "RGBA")

    # Polygon (nodes & edges)
    if ss.poly:
        # edges
        if len(ss.poly) >= 2:
            d.line(ss.poly, fill=(0,255,255,255), width=2)
        # closing edge preview if closed
        if ss.poly_closed and len(ss.poly) >= 3:
            d.line([ss.poly[-1], ss.poly[0]], fill=(0,255,255,255), width=2)
        # nodes
        for p in ss.poly:
            d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=(0,255,255,200))

    # Axes & joints
    def _draw_line(line, color):
        if len(line) == 2:
            d.line(line, fill=color, width=3)
            # endpoints
            for p in line:
                d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=color)

    _draw_line(ss.prox_axis, (66,133,244,255))
    _draw_line(ss.dist_axis, (221,0,221,255))
    _draw_line(ss.prox_joint, (255,215,0,220))
    _draw_line(ss.dist_joint, (255,215,0,220))

    # Points
    if ss.cora:
        x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)
    if ss.hinge:
        x,y=ss.hinge
        d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)

    # Report angles
    def _label_angle(line, label_y):
        if len(line) == 2:
            a = angle_deg(line[0], line[1])
            d.rectangle([6,label_y-12,220,label_y+6], fill=(0,0,0,160))
            d.text((10,label_y-10), f"angle {a:.1f}°", fill=(255,255,255,230))

    ytick=8
    if len(ss.prox_joint)==2: _label_angle(ss.prox_joint, ytick); ytick+=18
    if len(ss.dist_joint)==2: _label_angle(ss.dist_joint, ytick); ytick+=18
    if len(ss.prox_axis)==2:  _label_angle(ss.prox_axis,  ytick); ytick+=18
    if len(ss.dist_axis)==2:  _label_angle(ss.dist_axis,  ytick); ytick+=18

    return show

# --------------------------- session state -----------------------------

ss = st.session_state
defaults = dict(
    # basic
    dispw=1100,
    tool="Polygon",
    last_click=None,   # debouncer
    # geometry
    poly=[],
    poly_closed=False,
    hinge=None,
    cora=None,
    prox_axis=[],
    dist_axis=[],
    prox_joint=[],
    dist_joint=[],
    # sim
    move_segment="distal",   # "distal" or "proximal"
    dx=0, dy=0, theta=0,     # motion controls
)
for k,v in defaults.items(): ss.setdefault(k,v)

# ------------------------------ sidebar --------------------------------

st.sidebar.header("Load image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
ss.tool = st.sidebar.radio("Tool", ["Polygon", "Prox axis", "Dist axis",
                                    "Prox joint", "Dist joint", "HINGE", "CORA"],
                           index=["Polygon","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"].index(ss.tool))

# snappy UX helpers
c1,c2,c3,c4 = st.sidebar.columns(4)
if c1.button("Reset poly"):    ss.poly=[]; ss.poly_closed=False
if c2.button("Reset axes"):    ss.prox_axis=[]; ss.dist_axis=[]
if c3.button("Reset joints"):  ss.prox_joint=[]; ss.dist_joint=[]
if c4.button("Clear points"):  ss.hinge=None; ss.cora=None

ss.move_segment = st.sidebar.radio("Move which part after osteotomy?",
                                   ["distal","proximal"], horizontal=True,
                                   index=0 if ss.move_segment=="distal" else 1)

ss.dispw = st.sidebar.slider("Preview width", 600, 1800, ss.dispw, 50)

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

# ---------- apply osteotomy transform (if polygon closed) ----------
composite_for_display = disp.copy()

if ss.poly_closed and len(ss.poly) >= 3:
    # NOTE: polygon is in display (scaled) coordinates.
    poly_disp = ss.poly
    m_disp = polygon_mask(disp.size, poly_disp)
    inv_disp = ImageOps.invert(m_disp)

    prox_disp = Image.new("RGBA", disp.size, (0,0,0,0)); prox_disp.paste(disp, (0,0), inv_disp)
    dist_disp = Image.new("RGBA", disp.size, (0,0,0,0)); dist_disp.paste(disp, (0,0), m_disp)

    # Pick moving/fixed in display space
    moving = dist_disp if ss.move_segment=="distal" else prox_disp
    fixed  = prox_disp if ss.move_segment=="distal" else dist_disp

    # center to rotate about: prefer hinge, else centroid
    center = ss.hinge or centroid(poly_disp) or (disp.size[0]/2.0, disp.size[1]/2.0)

    moved = apply_affine_fragment(moving, ss.dx, ss.dy, ss.theta, center)

    # Composite
    base = Image.new("RGBA", disp.size, (0,0,0,0))
    base.alpha_composite(fixed)
    base.alpha_composite(moved)
    composite_for_display = base

# Draw overlays (lines/points) onto the current (possibly transformed) composite
overlay = draw_overlay(composite_for_display, ss)

# ---------- single image + immediate click handling ----------
# Show only once (this widget also captures clicks)
click = streamlit_image_coordinates(overlay.convert("RGB"), width=overlay.width, key="click-main")

# Debounce: recognise a *new* click only once
if click and "x" in click and "y" in click:
    event = (click["x"], click["y"], overlay.width, overlay.height, ss.tool)
    if event != ss.last_click:
        ss.last_click = event
        px, py = float(click["x"]), float(click["y"])

        # Map click into current display coords directly (overlay is display-sized)
        p = (px, py)

        # Tool behaviours (SNAPPY: immediate update)
        if ss.tool == "Polygon":
            if not ss.poly_closed:
                # close if near first
                if len(ss.poly) >= 3:
                    x0,y0 = ss.poly[0]
                    if (px-x0)**2 + (py-y0)**2 <= 10**2:
                        # snap close
                        ss.poly_closed = True
                    else:
                        ss.poly.append(p)
                else:
                    ss.poly.append(p)

        elif ss.tool == "Prox axis":
            if len(ss.prox_axis) < 1:
                ss.prox_axis = [p]
            elif len(ss.prox_axis) == 1:
                ss.prox_axis.append(p)
            else:
                # restart quickly
                ss.prox_axis = [p]

        elif ss.tool == "Dist axis":
            if len(ss.dist_axis) < 1:
                ss.dist_axis = [p]
            elif len(ss.dist_axis) == 1:
                ss.dist_axis.append(p)
            else:
                ss.dist_axis = [p]

        elif ss.tool == "Prox joint":
            if len(ss.prox_joint) < 1:
                ss.prox_joint = [p]
            elif len(ss.prox_joint) == 1:
                ss.prox_joint.append(p)
            else:
                ss.prox_joint = [p]

        elif ss.tool == "Dist joint":
            if len(ss.dist_joint) < 1:
                ss.dist_joint = [p]
            elif len(ss.dist_joint) == 1:
                ss.dist_joint.append(p)
            else:
                ss.dist_joint = [p]

        elif ss.tool == "HINGE":
            ss.hinge = p

        elif ss.tool == "CORA":
            ss.cora = p

        # After any click that changes geometry, re-run immediately (snappy)
        st.experimental_rerun()

# ------------- update axes positions when fragment moves -------------
# If polygon is closed & fragment moved, transform the axis that belongs to it
if ss.poly_closed and len(ss.poly) >= 3:
    center = ss.hinge or centroid(ss.poly) or (overlay.width/2.0, overlay.height/2.0)
    if ss.move_segment == "distal":
        if len(ss.dist_axis) == 2:
            ss.dist_axis = transform_points_y_down(ss.dist_axis, ss.dx, ss.dy, ss.theta, center)
        if len(ss.dist_joint) == 2:
            ss.dist_joint = transform_points_y_down(ss.dist_joint, ss.dx, ss.dy, ss.theta, center)
    else:
        if len(ss.prox_axis) == 2:
            ss.prox_axis = transform_points_y_down(ss.prox_axis, ss.dx, ss.dy, ss.theta, center)
        if len(ss.prox_joint) == 2:
            ss.prox_joint = transform_points_y_down(ss.prox_joint, ss.dx, ss.dy, ss.theta, center)

# -------------------- light footer with current tool -------------------
with st.expander("Status / help", expanded=False):
    st.write(f"**Tool**: {ss.tool}  |  Polygon closed: {ss.poly_closed}")
    st.write("Tip: double-clicking the first node while on Polygon closes the loop. "
             "When closed, pick HINGE (or it falls back to polygon centroid).")
