# app.py — single-image UI: draw & simulate in one place (no canvas)
import io, math
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy — single image", layout="wide")

# ---------- helpers ----------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def polygon_mask(size: Tuple[int,int], pts: List[Tuple[float,float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

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
    ax = p1a[0]-p0a[0]; ay = p1a[1]-p0a[1]
    bx = p1b[0]-p0b[0]; by = p1b[1]-p0b[1]
    La = math.hypot(ax,ay); Lb = math.hypot(bx,by)
    if La==0 or Lb==0: return None, None
    ax/=La; ay/=La; bx/=Lb; by/=Lb
    dot = max(-1.0, min(1.0, ax*bx + ay*by))
    theta = math.degrees(math.acos(dot))  # 0..180
    return theta, 180.0 - theta

def draw_label(im: Image.Image, xy, text):
    d = ImageDraw.Draw(im, "RGBA"); x,y = xy; pad=3
    box = d.textbbox((x,y), text)
    d.rectangle([x-pad,y-pad,x+box[2]-box[0]+pad,y+box[3]-box[1]+pad], fill=(0,0,0,140))
    d.text((x,y), text, fill=(255,255,255,235))

def transform_points_screen(points, dx, dy, angle_deg, center):
    if not points: return []
    ang = math.radians(angle_deg); c, s = math.cos(ang), math.sin(ang)
    cx, cy = center; out=[]
    for (x,y) in points:
        x0, y0 = x-cx, y-cy
        xr = x0*c + y0*s + cx + dx
        yr = -x0*s + y0*c + cy + dy
        out.append((float(xr), float(yr)))
    return out

def apply_osteotomy_overlay(src: Image.Image, poly: List[Tuple[float,float]], hinge: Tuple[float,float],
                            dx: float, dy: float, rot_deg: float, segment: str) -> Image.Image:
    """
    Overlay-style simulation: keep original visible; draw the moved segment on top.
    Nothing gets 'cut out'. If dx=dy=rot=0, result == original visually.
    """
    W,H = src.size
    m = polygon_mask((W,H), poly)
    inv = ImageOps.invert(m)
    inside = Image.new("RGBA", (W,H), (0,0,0,0)); inside.paste(src, (0,0), m)
    outside= Image.new("RGBA", (W,H), (0,0,0,0)); outside.paste(src,(0,0), inv)
    moving = inside if segment=="distal" else outside
    out = src.copy()  # base is the full image
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=hinge, expand=False)
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def dist2(a,b): return (a[0]-b[0])**2 + (a[1]-b[1])**2

# ---------- state ----------
ss = st.session_state
defaults = dict(
    dispw=1100,
    # tools: Joint line, Axis, Polygon, Hinge, Simulate
    tool="Joint line",
    # two joint tangent lines
    joint_lines={"proximal": None, "distal": None},  # {"p0":(x,y), "p1":(x,y)}
    placing_joint=None,        # {"segment":"proximal"/"distal","p0":(x,y)}
    joint_segment="distal",
    # axes with segment labels
    axes=[],                   # [{"segment":"proximal"/"distal","p0":(x,y),"p1":(x,y)}]
    placing_axis=None,         # {"segment":..., "p0":(x,y)}
    axis_segment="distal",
    # polygon + hinge
    poly=[],                   # [(x,y)...] original coords
    hinge=None,                # (x,y)
    # simulate controls (used live on single image)
    simulate_on=False,
    segment="distal",
    dx=0, dy=0, theta=0,
    click_nonce=0,
)
for k,v in defaults.items(): ss.setdefault(k, v)

# ---------- sidebar ----------
st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
if not up:
    st.info("Upload an image to begin.")
    st.stop()

st.sidebar.markdown("### Mode")
ss.tool = st.sidebar.radio("Tool", ["Joint line","Axis","Polygon","Hinge","Simulate"],
    index=["Joint line","Axis","Polygon","Hinge","Simulate"].index(ss.tool))

if ss.tool == "Joint line":
    ss.joint_segment = st.sidebar.radio("Joint line for", ["proximal","distal"],
        index=(0 if ss.joint_segment=="proximal" else 1), horizontal=True)
if ss.tool == "Axis":
    ss.axis_segment = st.sidebar.radio("New axis for", ["proximal","distal"],
        index=(0 if ss.axis_segment=="proximal" else 1), horizontal=True)

st.sidebar.markdown("### Display")
ss.dispw = st.sidebar.slider("Image width", 600, 1800, int(ss.dispw), 50)

st.sidebar.divider()
st.sidebar.markdown("### Simulation (live on single image)")
ss.simulate_on = st.sidebar.toggle("Simulation ON", value=bool(ss.simulate_on))
ss.segment = st.sidebar.radio("Move segment", ["distal","proximal"],
    index=(0 if ss.segment=="distal" else 1), horizontal=True)
ss.dx = st.sidebar.slider("ΔX (px)", -800, 800, int(ss.dx), 1)
ss.dy = st.sidebar.slider("ΔY (px)", -800, 800, int(ss.dy), 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, int(ss.theta), 1)

st.sidebar.divider()
st.sidebar.markdown("### Reset")
r1,r2,r3 = st.sidebar.columns(3)
if r1.button("Clear joints"): ss.joint_lines={"proximal":None,"distal":None}; ss.placing_joint=None
if r2.button("Clear axes"): ss.axes.clear(); ss.placing_axis=None
if r3.button("Clear poly/hinge"): ss.poly.clear(); ss.hinge=None

# ---------- image & scale ----------
src = load_rgba(up.getvalue())
W,H = src.size
scale = min(ss.dispw/float(W), 1.0)
dispW, dispH = int(round(W*scale)), int(round(H*scale))
def o2c(p): return (p[0]*scale, p[1]*scale)
def c2o(p): return (p[0]/scale, p[1]/scale)

# ---------- COMPOSE SINGLE DISPLAY IMAGE ----------
# If simulation ON and polygon closed + hinge set → overlay-style simulate
if ss.simulate_on and len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1] and ss.hinge:
    base = apply_osteotomy_overlay(src, ss.poly, ss.hinge, ss.dx, ss.dy, ss.theta, ss.segment)
else:
    base = src

disp = base.resize((dispW,dispH), Image.NEAREST).copy()
d = ImageDraw.Draw(disp, "RGBA")

# polygon wires/nodes
if ss.poly:
    ptsc=[o2c(p) for p in ss.poly]
    if len(ss.poly)>=2: d.line(ptsc, fill=(0,255,255,255), width=2)
    if len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1]:
        d.polygon(ptsc, outline=(0,255,255,255), fill=(0,255,255,35))
    for (x,y) in ptsc:
        d.ellipse([x-4,y-4,x+4,y+4], fill=(0,255,255,220))

# hinge marker
if ss.hinge:
    hx,hy=o2c(ss.hinge)
    d.ellipse([hx-7,hy-7,hx+7,hy+7], outline=(255,165,0,255), width=3)
    d.line([(hx-12,hy),(hx+12,hy)], fill=(255,165,0,255), width=1)
    d.line([(hx,hy-12),(hx,hy+12)], fill=(255,165,0,255), width=1)

# joint lines (both)
joint_colors = {"proximal": (0,255,255,255), "distal": (255,105,180,255)}  # cyan / pink
for seg in ("proximal","distal"):
    jl = ss.joint_lines.get(seg)
    if jl:
        j0d, j1d = o2c(jl["p0"]), o2c(jl["p1"])
        a,b = extend_inf_line_through_image(j0d, j1d, dispW, dispH)
        d.line([a,b], fill=joint_colors[seg], width=2)
        for q in [j0d,j1d]:
            d.ellipse([q[0]-4,q[1]-4,q[0]+4,q[1]+4], fill=joint_colors[seg])

# axes — and if simulating, transform the moving segment's axes so they "stick"
angles_sidebar = []
for ax in ss.axes:
    p0,p1 = ax["p0"], ax["p1"]

    if ss.simulate_on and len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1] and ss.hinge:
        if ax["segment"] == ss.segment:
            p0,p1 = transform_points_screen([p0,p1], ss.dx, ss.dy, ss.theta, ss.hinge)
            p0,p1 = p0[0], p1[0]  # unpack

    col = (66,133,244,255) if ax["segment"]=="proximal" else (221,0,221,255)
    d.line([o2c(p0), o2c(p1)], fill=col, width=3)
    d.ellipse([o2c(p0)[0]-4,o2c(p0)[1]-4,o2c(p0)[0]+4,o2c(p0)[1]+4], fill=(255,255,255,220))
    d.ellipse([o2c(p1)[0]-4,o2c(p1)[1]-4,o2c(p1)[0]+4,o2c(p1)[1]+4], fill=(255,255,255,220))

    # angle vs matching joint line (if present)
    jl = ss.joint_lines.get(ax["segment"])
    if jl:
        j0d, j1d = o2c(jl["p0"]), o2c(jl["p1"])
        acute, obtuse = angle_between_lines_deg(j0d, j1d, o2c(p0), o2c(p1))
        if acute is not None:
            mid = ((o2c(p0)[0]+o2c(p1)[0])/2, (o2c(p0)[1]+o2c(p1)[1])/2)
            draw_label(disp, (mid[0]+6, mid[1]+6), f"{acute:.1f}°/{obtuse:.1f}°")
            angles_sidebar.append((ax["segment"], acute, obtuse))

# ---------- SHOW SINGLE IMAGE & CAPTURE CLICKS ----------
st.image(disp, use_column_width=False, width=dispW)
click = streamlit_image_coordinates(disp.convert("RGB"), width=dispW, key=f"click-{ss.click_nonce}")
xo=yo=None
if click and "x" in click and "y" in click:
    xo,yo = c2o((float(click["x"]), float(click["y"])))

def toast(msg): st.toast(msg)

# ---------- CLICK HANDLERS (single-click, no rerun forcing) ----------
if ss.tool == "Joint line":
    if xo is not None:
        if ss.placing_joint is None:
            ss.placing_joint = {"segment": ss.joint_segment, "p0": (xo,yo)}
            toast(f"{ss.joint_segment.capitalize()} joint: first point set")
        else:
            seg = ss.placing_joint["segment"]
            ss.joint_lines[seg] = {"p0": ss.placing_joint["p0"], "p1": (xo,yo)}
            ss.placing_joint = None
            toast(f"{seg.capitalize()} joint line set")

elif ss.tool == "Axis":
    if xo is not None:
        if ss.placing_axis is None:
            ss.placing_axis = {"segment": ss.axis_segment, "p0": (xo,yo)}
            toast(f"{ss.axis_segment.capitalize()} axis origin set")
        else:
            seg = ss.placing_axis["segment"]
            ss.axes.append({"segment": seg, "p0": ss.placing_axis["p0"], "p1": (xo,yo)})
            ss.placing_axis = None
            toast(f"{seg.capitalize()} axis added")

elif ss.tool == "Polygon":
    # small inline controls to avoid confusion
    t1,t2,t3,t4 = st.columns([1,1,1,3])
    if t1.button("Undo"): 
        if ss.poly: ss.poly.pop(); toast("Polygon point removed")
    if t2.button("Clear"): 
        ss.poly.clear(); toast("Polygon cleared")
    if t3.button("Close"):
        if len(ss.poly)>=3:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
    if xo is not None:
        if len(ss.poly)>=2 and math.hypot(xo-ss.poly[0][0], yo-ss.poly[0][1]) < 12/scale:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
        else:
            ss.poly.append((xo,yo)); toast("Point added")

elif ss.tool == "Hinge":
    if xo is not None:
        ss.hinge = (xo,yo); toast("Hinge set")

# ---------- sidebar angle list ----------
if angles_sidebar:
    st.sidebar.markdown("### Axis ∠ vs joint line")
    for i,(seg,a,o) in enumerate(angles_sidebar,1):
        st.sidebar.write(f"{seg.capitalize()} axis {i}: **{a:.1f}° / {o:.1f}°**")
