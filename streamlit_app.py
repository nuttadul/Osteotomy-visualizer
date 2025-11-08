# app.py — Angle-assisted axes with joint line (tangent or freehand), axis origin choice, and ghost line
import io, math
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Angle-assisted Axes (Bone-Ninja style)", layout="wide")

# --------------------- helpers ---------------------
def load_rgba(file_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(file_bytes))
    return ImageOps.exif_transpose(im).convert("RGBA")

def line_angle(p0, p1) -> float:
    """Angle (radians) of vector p0->p1 in screen coords (y down)."""
    return math.atan2(p1[1]-p0[1], p1[0]-p0[0])

def extend_inf_line_through_image(p0, p1, w, h) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """Return endpoints where infinite line through p0-p1 intersects image bounds."""
    x0,y0 = p0; x1,y1 = p1
    dx, dy = x1-x0, y1-y0
    eps = 1e-9
    if abs(dx) < eps and abs(dy) < eps:
        return p0, p1
    ts = []
    # Intersections with x=0 and x=w
    if abs(dx) > eps:
        t = (0 - x0)/dx; y = y0 + t*dy
        if 0 <= y <= h: ts.append(t)
        t = (w - x0)/dx; y = y0 + t*dy
        if 0 <= y <= h: ts.append(t)
    # Intersections with y=0 and y=h
    if abs(dy) > eps:
        t = (0 - y0)/dy; x = x0 + t*dx
        if 0 <= x <= w: ts.append(t)
        t = (h - y0)/dy; x = x0 + t*dx
        if 0 <= x <= w: ts.append(t)
    if len(ts) < 2:
        return p0, p1
    t0, t1 = min(ts), max(ts)
    a = (x0 + t0*dx, y0 + t0*dy)
    b = (x0 + t1*dx, y0 + t1*dy)
    return a, b

def draw_grid(img: Image.Image, step: int = 50, alpha: int = 40) -> Image.Image:
    im = img.copy()
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    col = (255,255,255,alpha)
    for x in range(0, w, step):
        d.line([(x,0),(x,h)], fill=col, width=1)
    for y in range(0, h, step):
        d.line([(0,y),(w,y)], fill=col, width=1)
    return im

def draw_crosshair(im: Image.Image, p: Tuple[float,float], size=18, alpha=130):
    d = ImageDraw.Draw(im, "RGBA")
    x,y = p
    col = (255,255,0,alpha)
    d.line([(x-size,y),(x+size,y)], fill=col, width=1)
    d.line([(x,y-size),(x,y+size)], fill=col, width=1)

def to_disp(p, scale, zoom): return (p[0]*scale*zoom, p[1]*scale*zoom)
def to_orig(p, scale, zoom): return (p[0]/(scale*zoom), p[1]/(scale*zoom))

# --------------------- state ---------------------
ss = st.session_state
defaults = dict(
    # image & view
    dispw=1100, zoom=1.0, show_grid=False,
    # joint drawing
    joint_mode="Tangent (2 clicks)",   # or "Freehand (multi-click)"
    joint_pts=[],                      # in ORIGINAL pixels
    # axis controls
    axis_angle_deg=81.0,               # angle relative to joint line
    axis_origin_mode="Joint center",   # or "Click to set"
    axis_origin=None,                  # if "Click to set", stored in ORIGINAL pixels
    axis_line=[],                      # resulting axis line (p0,p1) in ORIGINAL pixels
    # ghost line
    guide_on=True, guide_snap5=True, guide_len=400, guide_angle_offset=0.0,
)
for k,v in defaults.items(): ss.setdefault(k, v)

# --------------------- sidebar ---------------------
st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
if not up:
    st.info("Upload an X-ray to begin.")
    st.stop()

ss.dispw = st.sidebar.slider("Preview width", 600, 1800, int(ss.dispw), 50)
ss.zoom  = st.sidebar.slider("Zoom", 0.5, 4.0, float(ss.zoom), 0.1)
ss.show_grid = st.sidebar.toggle("Show grid", value=ss.show_grid)

st.sidebar.divider()
st.sidebar.subheader("Joint line")
ss.joint_mode = st.sidebar.radio("Mode", ["Tangent (2 clicks)", "Freehand (multi-click)"], index=(0 if ss.joint_mode.startswith("Tangent") else 1))
col1,col2 = st.sidebar.columns(2)
if col1.button("Clear joint"): ss.joint_pts.clear()
if col2.button("Finish freehand"):
    if ss.joint_mode.startswith("Freehand") and len(ss.joint_pts) >= 2:
        pass  # nothing extra needed; line will be extended automatically

st.sidebar.divider()
st.sidebar.subheader("Axis construction")
ss.axis_angle_deg = st.sidebar.number_input("Angle relative to joint (°)", 0.0, 180.0, float(ss.axis_angle_deg), 0.5)
ss.axis_origin_mode = st.sidebar.radio("Axis origin", ["Joint center","Click to set"], index=(0 if ss.axis_origin_mode=="Joint center" else 1))
if st.sidebar.button("Clear axis"):
    ss.axis_line.clear()
    ss.axis_origin = None

st.sidebar.divider()
st.sidebar.subheader("Ghost / Guide")
ss.guide_on = st.sidebar.toggle("Enable ghost line", value=ss.guide_on)
ss.guide_snap5 = st.sidebar.toggle("Snap guide to 5°", value=ss.guide_snap5)
ss.guide_angle_offset = st.sidebar.slider("Guide angle offset (°)", -180.0, 180.0, float(ss.guide_angle_offset), 0.5)
ss.guide_len = st.sidebar.slider("Guide length (px)", 50, 2000, int(ss.guide_len), 10)

# --------------------- image & scale ---------------------
src = load_rgba(up.getvalue())
W,H = src.size
scale = min(ss.dispw/float(W), 1.0)
dispW, dispH = int(round(W*scale*ss.zoom)), int(round(H*scale*ss.zoom))
disp = src.resize((dispW, dispH), Image.NEAREST)
if ss.show_grid:
    disp = draw_grid(disp, step=max(20, int(50*ss.zoom)))

# --------------------- build a preview layer (overlays) ---------------------
overlay = disp.copy()
d = ImageDraw.Draw(overlay, "RGBA")

# 1) draw joint line (either tangent or freehand)
joint_has_line = False
joint_line_disp = None   # display-space line segment across image bounds
if len(ss.joint_pts) >= 2:
    # Use first and last for direction; extend across entire image
    p0, p1 = ss.joint_pts[0], ss.joint_pts[-1]
    a, b = extend_inf_line_through_image(
        to_disp(p0, scale, ss.zoom), to_disp(p1, scale, ss.zoom), dispW, dispH
    )
    joint_has_line = True
    joint_line_disp = (a,b)
    d.line([a,b], fill=(0,255,255,255), width=2)  # cyan full-width joint line
    # draw the clicked points too
    ptsd = [to_disp(p, scale, ss.zoom) for p in ss.joint_pts]
    for q in ptsd:
        d.ellipse([q[0]-3,q[1]-3,q[0]+3,q[1]+3], fill=(0,255,255,200))

# 2) axis origin logic
axis_origin_disp = None
if ss.axis_origin_mode == "Joint center":
    if joint_has_line:
        # midpoint of the two display endpoints, convert back to original for storage
        (ax0,ay0),(ax1,ay1) = joint_line_disp
        mid_disp = ((ax0+ax1)/2.0, (ay0+ay1)/2.0)
        axis_origin_disp = mid_disp
        ss.axis_origin = to_orig(mid_disp, scale, ss.zoom)
else:
    # "Click to set" — user will click on the image; if set, draw a crosshair
    if ss.axis_origin is not None:
        axis_origin_disp = to_disp(ss.axis_origin, scale, ss.zoom)

# 3) ghost / guide line
def draw_ghost(origin_disp, base_angle_rad):
    """Draw ghost from origin with angle=base±offset and length guide_len."""
    if not ss.guide_on: return
    offset = ss.guide_angle_offset
    if ss.guide_snap5:
        offset = round(offset/5.0)*5.0
    th = base_angle_rad - math.radians(offset)  # clockwise positive (screen coords)
    x0,y0 = origin_disp
    x1 = x0 + ss.guide_len * math.cos(th)
    y1 = y0 + ss.guide_len * math.sin(th)
    d.line([(x0,y0),(x1,y1)], fill=(255,200,0,220), width=2)
    draw_crosshair(overlay, origin_disp, size=16, alpha=180)
    # small HUD
    d.rectangle([10, 10, 280, 58], fill=(0,0,0,150))
    txt = f"guide angle offset {offset:.1f}°, length {ss.guide_len}px"
    d.text((18,20), txt, fill=(255,255,255,240))

# base angle is joint line angle
if joint_has_line:
    (ja0,ja1) = joint_line_disp
    base_ang = math.atan2(ja1[1]-ja0[1], ja1[0]-ja0[0])      # radians, display space
else:
    base_ang = 0.0

# axis construction angle: “axis_angle_deg” relative to the joint line (normal/oblique)
# axis direction angle in display space:
axis_dir_disp_rad = base_ang - math.radians(ss.axis_angle_deg)

# draw ghost if possible
if axis_origin_disp is not None:
    draw_ghost(axis_origin_disp, axis_dir_disp_rad)

# --------------------- click handling ---------------------
# (we capture clicks on the overlay image with visual guides drawn)
click = streamlit_image_coordinates(overlay, width=dispW, key="click-main")

if click and "x" in click and "y" in click:
    p_disp = (float(click["x"]), float(click["y"]))
    p = to_orig(p_disp, scale, ss.zoom)

    # If axis origin mode is "Click to set", set/replace it
    if ss.axis_origin_mode == "Click to set":
        ss.axis_origin = p

    # Joint drawing modes
    if ss.joint_mode.startswith("Tangent"):
        # Two clicks define tangent; third click restarts
        if len(ss.joint_pts) == 0:
            ss.joint_pts = [p]
        elif len(ss.joint_pts) == 1:
            ss.joint_pts.append(p)
        else:
            ss.joint_pts = [p]
    else:
        # Freehand (multi-click). Each click adds a vertex; finish with the button.
        ss.joint_pts.append(p)

    # Once we have a joint line and an origin, build the axis line immediately
    if len(ss.joint_pts) >= 2 and ss.axis_origin is not None:
        # axis from origin at axis_dir (computed in DISPLAY), so convert to ORIGINAL
        # length: span image; make a long segment and then extend to bounds
        origin_disp = to_disp(ss.axis_origin, scale, ss.zoom)
        x0,y0 = origin_disp
        x1 = x0 + ss.guide_len * math.cos(axis_dir_disp_rad)
        y1 = y0 + ss.guide_len * math.sin(axis_dir_disp_rad)
        # extend across display image bounds
        a_disp, b_disp = extend_inf_line_through_image((x0,y0),(x1,y1), dispW, dispH)
        # store in ORIGINAL pixels
        ss.axis_line = [to_orig(a_disp, scale, ss.zoom), to_orig(b_disp, scale, ss.zoom)]

# --------------------- final preview ---------------------
final = disp.copy()
fd = ImageDraw.Draw(final, "RGBA")

# redraw joint (full width)
if joint_line_disp is not None:
    fd.line(list(joint_line_disp), fill=(0,255,255,255), width=2)

# redraw joint clicks
for q in [to_disp(p, scale, ss.zoom) for p in ss.joint_pts]:
    fd.ellipse([q[0]-3,q[1]-3,q[0]+3,q[1]+3], fill=(0,255,255,200))

# axis origin marker
if ss.axis_origin is not None:
    draw_crosshair(final, to_disp(ss.axis_origin, scale, ss.zoom), size=18, alpha=200)

# axis line
if len(ss.axis_line) == 2:
    fd.line([to_disp(ss.axis_line[0], scale, ss.zoom), to_disp(ss.axis_line[1], scale, ss.zoom)],
            fill=(66,133,244,255), width=3)

st.image(final, width=dispW, caption="Angle-assisted axes (joint + axis)")

st.caption("Tips: Choose 'Tangent (2 clicks)' for fast joint line, 'Freehand' for curved/irregular joints. "
           "Pick axis origin as 'Joint center' or click to set. The ghost line shows the axis direction before committing.")
