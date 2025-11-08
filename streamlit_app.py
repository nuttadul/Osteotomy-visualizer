# app.py — Joint tangent, freehand axes with angle feedback, and distal axis that follows osteotomy
import io, math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy: Joint + Axes + Osteotomy", layout="wide")

# ----------------------- helpers -----------------------
def load_rgba(file_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(file_bytes))
    return ImageOps.exif_transpose(im).convert("RGBA")

def polygon_mask(size: Tuple[int,int], pts: List[Tuple[float,float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def apply_osteotomy(src: Image.Image, poly: List[Tuple[float,float]], hinge: Tuple[float,float],
                    dx: float, dy: float, rot_deg: float, segment: str) -> Image.Image:
    W,H = src.size
    m = polygon_mask((W,H), poly)
    inv = ImageOps.invert(m)
    inside = Image.new("RGBA", (W,H), (0,0,0,0)); inside.paste(src, (0,0), m)
    outside= Image.new("RGBA", (W,H), (0,0,0,0)); outside.paste(src,(0,0), inv)
    moving = inside if segment=="distal" else outside
    fixed  = outside if segment=="distal" else inside
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=hinge, expand=False)
    out = Image.new("RGBA", (W,H), (0,0,0,0))
    out.alpha_composite(fixed, (0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def transform_points_screen(points, dx, dy, angle_deg, center):
    if not points: return []
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    cx, cy = center
    out = []
    for (x,y) in points:
        x0, y0 = x - cx, y - cy
        xr = x0*c + y0*s + cx + dx
        yr = -x0*s + y0*c + cy + dy
        out.append((float(xr), float(yr)))
    return out

def extend_inf_line_through_image(p0, p1, w, h):
    x0,y0 = p0; x1,y1 = p1
    dx, dy = x1-x0, y1-y0
    eps = 1e-9
    if abs(dx) < eps and abs(dy) < eps:
        return p0, p1
    ts = []
    if abs(dx) > eps:
        t = (0 - x0)/dx; y = y0 + t*dy
        if 0 <= y <= h: ts.append(t)
        t = (w - x0)/dx; y = y0 + t*dy
        if 0 <= y <= h: ts.append(t)
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

def angle_between_lines_deg(p0a,p1a,p0b,p1b):
    """Return unsigned acute angle (0..180) between two lines."""
    ax = p1a[0]-p0a[0]; ay = p1a[1]-p0a[1]
    bx = p1b[0]-p0b[0]; by = p1b[1]-p0b[1]
    La = math.hypot(ax,ay); Lb = math.hypot(bx,by)
    if La==0 or Lb==0: return None, None
    ax/=La; ay/=La; bx/=Lb; by/=Lb
    dot = max(-1.0, min(1.0, ax*bx + ay*by))
    theta = math.degrees(math.acos(dot))  # 0..180
    return theta, 180.0 - theta           # return both acute and obtuse

def draw_small_label(im: Image.Image, xy: Tuple[float,float], text: str):
    """Tiny unobtrusive label on image."""
    d = ImageDraw.Draw(im, "RGBA")
    x,y = xy
    pad = 3
    bbox = d.textbbox((x,y), text)
    bw = bbox[2]-bbox[0]; bh = bbox[3]-bbox[1]
    box = [x-pad, y-pad, x+bw+pad, y+bh+pad]
    d.rectangle(box, fill=(0,0,0,120))
    d.text((x,y), text, fill=(255,255,255,230))

def dist2(a,b): return (a[0]-b[0])**2 + (a[1]-b[1])**2

# ----------------------- state -----------------------
ss = st.session_state
defaults = dict(
    dispw=1100,
    tool="Joint line",                           # "Joint line", "Axis", "Polygon", "Hinge", "Simulate"
    joint_line=None,                             # {"p0":(x,y), "p1":(x,y)} in original px
    placing_joint=None,                          # (x,y) while waiting second click
    axes=[],                                     # [{"p0":(x,y), "p1":(x,y), "segment":"distal"|"proximal"}]
    placing_axis=None,                           # {"p0":(x,y), "segment":...}
    axis_segment="distal",                       # default segment for new axis
    poly=[],                                     # polygon points (original)
    hinge=None,                                  # (x,y)
    # simulate
    segment="distal", dx=0, dy=0, theta=0,
    click_nonce=0,
)
for k,v in defaults.items(): ss.setdefault(k, v)

# ----------------------- sidebar -----------------------
st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
if not up:
    st.info("Upload an X-ray to begin.")
    st.stop()

st.sidebar.markdown("### Tools")
ss.tool = st.sidebar.radio("Mode", ["Joint line","Axis","Polygon","Hinge","Simulate"], index=["Joint line","Axis","Polygon","Hinge","Simulate"].index(ss.tool))

if ss.tool == "Axis":
    st.sidebar.markdown("**New axis belongs to**")
    ss.axis_segment = st.sidebar.radio("Segment", ["distal","proximal"], index=(0 if ss.axis_segment=="distal" else 1), horizontal=True)

st.sidebar.divider()
st.sidebar.markdown("### Reset")
c1,c2,c3 = st.sidebar.columns(3)
if c1.button("Clear joint"):
    ss.joint_line=None; ss.placing_joint=None
if c2.button("Clear axes"):
    ss.axes.clear(); ss.placing_axis=None
if c3.button("Clear poly/hinge"):
    ss.poly.clear(); ss.hinge=None

st.sidebar.divider()
st.sidebar.markdown("### Simulate (quick)")
ss.segment = st.sidebar.radio("Move", ["distal","proximal"], index=(0 if ss.segment=="distal" else 1), horizontal=True)
ss.dx = st.sidebar.slider("ΔX (px)", -500, 500, int(ss.dx), 1)
ss.dy = st.sidebar.slider("ΔY (px)", -500, 500, int(ss.dy), 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, int(ss.theta), 1)

# ----------------------- image/scale -----------------------
src = load_rgba(up.getvalue())
W,H = src.size
scale = min(ss.dispw/float(W), 1.0)
dispW, dispH = int(round(W*scale)), int(round(H*scale))

def o2c(p): return (p[0]*scale, p[1]*scale)
def c2o(p): return (p[0]/scale, p[1]/scale)

# ----------------------- layout -----------------------
left, right = st.columns([1.05,1])
left.subheader("Live drawing")
right.subheader("Preview / Simulation")

# ----------------------- overlay (left) -----------------------
overlay = src.resize((dispW,dispH), Image.NEAREST).copy()
d = ImageDraw.Draw(overlay, "RGBA")

# draw joint line (extended across image)
joint_disp = None
if ss.joint_line:
    j0d, j1d = o2c(ss.joint_line["p0"]), o2c(ss.joint_line["p1"])
    a,b = extend_inf_line_through_image(j0d, j1d, dispW, dispH)
    d.line([a,b], fill=(0,255,255,255), width=2)  # cyan
    # nodes
    for q in [j0d, j1d]:
        d.ellipse([q[0]-4,q[1]-4,q[0]+4,q[1]+4], fill=(0,255,255,200))
    joint_disp = (a,b)

# draw polygon (wireframe + nodes)
if ss.poly:
    ptsc = [o2c(p) for p in ss.poly]
    if len(ss.poly)>=2:
        d.line(ptsc, fill=(0,255,255,255), width=2)
    if len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1]:
        d.polygon(ptsc, outline=(0,255,255,255), fill=(0,255,255,35))
    for q in ptsc:
        d.ellipse([q[0]-4,q[1]-4,q[0]+4,q[1]+4], fill=(0,255,255,200))

# draw axes (and angle labels if joint exists)
angles_sidebar = []
for ax in ss.axes:
    p0d, p1d = o2c(ax["p0"]), o2c(ax["p1"])
    d.line([p0d,p1d], fill=(66,133,244,255) if ax["segment"]=="proximal" else (221,0,221,255), width=3)
    d.ellipse([p0d[0]-4,p0d[1]-4,p0d[0]+4,p0d[1]+4], fill=(255,255,255,200))
    d.ellipse([p1d[0]-4,p1d[1]-4,p1d[0]+4,p1d[1]+4], fill=(255,255,255,200))
    if joint_disp:
        acute, obtuse = angle_between_lines_deg(joint_disp[0], joint_disp[1], p0d, p1d)
        if acute is not None:
            mid = ((p0d[0]+p1d[0])/2, (p0d[1]+p1d[1])/2)
            draw_small_label(overlay, (mid[0]+6, mid[1]+6), f"{acute:.1f}° / {obtuse:.1f}°")
            angles_sidebar.append((ax["segment"], acute, obtuse))

# hinge
if ss.hinge:
    hx,hy = o2c(ss.hinge)
    d.ellipse([hx-7,hy-7,hx+7,hy+7], outline=(255,165,0,255), width=3)
    d.line([(hx-12,hy),(hx+12,hy)], fill=(255,165,0,255), width=1)
    d.line([(hx,hy-12),(hx,hy+12)], fill=(255,165,0,255), width=1)

# show angles in sidebar
if angles_sidebar:
    st.sidebar.markdown("### Axis ∠ vs joint")
    for i,(seg,a,o) in enumerate(angles_sidebar,1):
        st.sidebar.write(f"Axis {i} ({seg}): **{a:.1f}° / {o:.1f}°**")

# capture click (RGB to avoid JPEG alpha issue)
click = streamlit_image_coordinates(overlay.convert("RGB"), width=dispW, key=f"click-{ss.click_nonce}")
xo=yo=None
if click and "x" in click and "y" in click:
    xo,yo = c2o((float(click["x"]), float(click["y"])))

# ----------------------- click handling -----------------------
def toast(msg): st.toast(msg)

if ss.tool == "Joint line":
    # two clicks define the tangent; extended for drawing
    if xo is not None:
        if ss.placing_joint is None:
            ss.placing_joint = (xo,yo)
            toast("Joint: first point set")
        else:
            ss.joint_line = {"p0": ss.placing_joint, "p1": (xo,yo)}
            ss.placing_joint = None
            toast("Joint line set")

elif ss.tool == "Axis":
    # freehand axis: two clicks, belongs to chosen segment (distal/proximal)
    if xo is not None:
        if ss.placing_axis is None:
            ss.placing_axis = {"p0":(xo,yo), "segment": ss.axis_segment}
            toast(f"Axis origin set ({ss.axis_segment})")
        else:
            p0 = ss.placing_axis["p0"]
            ss.axes.append({"p0":p0, "p1":(xo,yo), "segment": ss.placing_axis["segment"]})
            ss.placing_axis = None
            toast("Axis added")

elif ss.tool == "Polygon":
    cols = left.columns([1,1,1,1])
    if cols[0].button("Undo last"):
        if ss.poly: ss.poly.pop(); toast("Polygon point removed")
    if cols[1].button("Clear poly"):
        ss.poly.clear(); toast("Polygon cleared")
    if cols[2].button("Close poly"):
        if len(ss.poly)>=3:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
    cols[3].markdown("&nbsp;")
    if xo is not None:
        if len(ss.poly)>=2 and math.hypot(xo-ss.poly[0][0], yo-ss.poly[0][1]) < 12/scale:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
        else:
            ss.poly.append((xo,yo)); toast("Point added")

elif ss.tool == "Hinge":
    if xo is not None:
        ss.hinge = (xo,yo); toast("Hinge set")

# ----------------------- preview / simulate (right) -----------------------
# Start with source or transformed image depending on simulate intent
preview = src
move_now = (ss.tool == "Simulate")
if move_now and len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1] and ss.hinge:
    preview = apply_osteotomy(src, ss.poly, ss.hinge, ss.dx, ss.dy, ss.theta, ss.segment)

# Draw overlays again on preview (axes must follow their assigned segment)
disp = preview.resize((dispW,dispH), Image.NEAREST).copy()
dr = ImageDraw.Draw(disp, "RGBA")

# joint line on preview
if ss.joint_line:
    j0d,j1d = o2c(ss.joint_line["p0"]), o2c(ss.joint_line["p1"])
    a,b = extend_inf_line_through_image(j0d, j1d, dispW, dispH)
    dr.line([a,b], fill=(0,255,255,255), width=2)
    for q in [j0d,j1d]:
        dr.ellipse([q[0]-4,q[1]-4,q[0]+4,q[1]+4], fill=(0,255,255,200))

# polygon on preview
if ss.poly:
    ptsc=[o2c(p) for p in ss.poly]
    if len(ss.poly)>=2:
        dr.line(ptsc, fill=(0,255,255,255), width=2)
    if len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1]:
        dr.polygon(ptsc, outline=(0,255,255,255), fill=(0,255,255,35))

# axes on preview (apply the SAME transform to endpoints as the image segment)
for ax in ss.axes:
    p0, p1 = ax["p0"], ax["p1"]
    if move_now and len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1] and ss.hinge:
        if ax["segment"] == ss.segment:
            # axis belongs to the moving segment → transform its endpoints
            p0, p1 = transform_points_screen([p0,p1], ss.dx, ss.dy, ss.theta, ss.hinge)
            p0, p1 = p0[0], p1[1]  # unpack result pairs
            # the above returns list; careful:
            p0, p1 = transform_points_screen([ax["p0"], ax["p1"]], ss.dx, ss.dy, ss.theta, ss.hinge)
            p0, p1 = p0[0], p1[0]  # correct unpack
    col = (66,133,244,255) if ax["segment"]=="proximal" else (221,0,221,255)
    dr.line([o2c(p0), o2c(p1)], fill=col, width=3)
    dr.ellipse([o2c(p0)[0]-4,o2c(p0)[1]-4,o2c(p0)[0]+4,o2c(p0)[1]+4], fill=(255,255,255,200))
    dr.ellipse([o2c(p1)[0]-4,o2c(p1)[1]-4,o2c(p1)[0]+4,o2c(p1)[1]+4], fill=(255,255,255,200))

# hinge on preview
if ss.hinge:
    hx,hy = o2c(ss.hinge)
    dr.ellipse([hx-7,hy-7,hx+7,hy+7], outline=(255,165,0,255), width=3)
    dr.line([(hx-12,hy),(hx+12,hy)], fill=(255,165,0,255), width=1)
    dr.line([(hx,hy-12),(hx,hy+12)], fill=(255,165,0,255), width=1)

right.image(disp, use_column_width=False, width=dispW)
