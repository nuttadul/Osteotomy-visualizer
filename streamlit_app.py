# app.py — clean aligned UI, click nodes, proximal/distal joint & axes, angles, and osteotomy with sticky distal axes
import io, math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – joints, axes & osteotomy (click mode)", layout="wide")

# ---------------- helpers ----------------
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
    d = ImageDraw.Draw(im, "RGBA")
    x,y = xy; pad=3
    box = d.textbbox((x,y), text)
    d.rectangle([x-pad,y-pad,x+box[2]-box[0]+pad,y+box[3]-box[1]+pad], fill=(0,0,0,140))
    d.text((x,y), text, fill=(255,255,255,235))

def transform_points_screen(points, dx, dy, angle_deg, center):
    if not points: return []
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    cx, cy = center
    out = []
    for (x, y) in points:
        x0, y0 = x - cx, y - cy
        xr = x0*c + y0*s + cx + dx
        yr = -x0*s + y0*c + cy + dy
        out.append((float(xr), float(yr)))
    return out

def apply_osteotomy_overlay(src: Image.Image, poly: List[Tuple[float,float]], hinge: Tuple[float,float],
                            dx: float, dy: float, rot_deg: float, segment: str) -> Image.Image:
    """
    Overlay-style: start from the original full image, then draw the moved segment on top.
    Nothing disappears; the moving half overlays the base.
    """
    W,H = src.size
    m = polygon_mask((W,H), poly)
    inv = ImageOps.invert(m)
    inside = Image.new("RGBA", (W,H), (0,0,0,0)); inside.paste(src, (0,0), m)
    outside= Image.new("RGBA", (W,H), (0,0,0,0)); outside.paste(src,(0,0), inv)

    moving = inside if segment=="distal" else outside
    # base = original image
    out = src.copy()
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=hinge, expand=False)
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def dist2(a,b): return (a[0]-b[0])**2 + (a[1]-b[1])**2

# --------------- state ---------------
ss = st.session_state
defaults = dict(
    dispw=1100,
    tool="Joint line",        # "Joint line", "Axis", "Polygon", "Hinge", "Simulate"
    # joint lines for both segments
    joint_lines={"proximal": None, "distal": None},   # {"p0":(x,y), "p1":(x,y)}
    placing_joint=None,       # {"segment": "proximal"/"distal", "p0":(x,y)}
    joint_segment="distal",   # default segment to place joint for
    # axes (freehand) with segment tag
    axes=[],                  # [{"segment":"proximal"/"distal", "p0":(x,y), "p1":(x,y)}]
    placing_axis=None,        # {"segment": ..., "p0":(x,y)}
    axis_segment="distal",    # which segment a new axis belongs to
    # polygon & hinge
    poly=[],                  # polygon points (original px)
    hinge=None,               # (x,y)
    # simulate
    segment="distal", dx=0, dy=0, theta=0,
    click_nonce=0,
)
for k,v in defaults.items(): ss.setdefault(k, v)

# --------------- sidebar ---------------
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
                                        index=(0 if ss.joint_segment=="proximal" else 1),
                                        horizontal=True)
if ss.tool == "Axis":
    ss.axis_segment = st.sidebar.radio("New axis for", ["proximal","distal"],
                                       index=(0 if ss.axis_segment=="proximal" else 1),
                                       horizontal=True)

st.sidebar.markdown("### Display")
ss.dispw = st.sidebar.slider("Preview width", 600, 1800, int(ss.dispw), 50)

st.sidebar.divider()
st.sidebar.markdown("### Reset")
c1,c2,c3 = st.sidebar.columns(3)
if c1.button("Clear joints"):
    ss.joint_lines={"proximal": None, "distal": None}; ss.placing_joint=None
if c2.button("Clear axes"):
    ss.axes.clear(); ss.placing_axis=None
if c3.button("Clear poly/hinge"):
    ss.poly.clear(); ss.hinge=None

st.sidebar.divider()
st.sidebar.markdown("### Quick simulate (same sliders used in Simulate tool)")
ss.segment = st.sidebar.radio("Move segment", ["distal","proximal"],
                              index=(0 if ss.segment=="distal" else 1), horizontal=True)
ss.dx = st.sidebar.slider("ΔX (px)", -800, 800, int(ss.dx), 1)
ss.dy = st.sidebar.slider("ΔY (px)", -800, 800, int(ss.dy), 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, int(ss.theta), 1)

# --------------- image/scale ---------------
src = load_rgba(up.getvalue())
W,H = src.size
scale = min(ss.dispw/float(W), 1.0)
dispW, dispH = int(round(W*scale)), int(round(H*scale))

def o2c(p): return (p[0]*scale, p[1]*scale)
def c2o(p): return (p[0]/scale, p[1]/scale)

# --------------- aligned layout (same level) ---------------
left, right = st.columns(2, vertical_alignment="top")  # ensures same top alignment

# a slim caption instead of tall headers keeps images level-aligned
left.caption("Live drawing (click once to add a node; nodes appear immediately)")
right.caption("Preview / Simulation")

# --------------- LIVE WINDOW (left) ---------------
# Start from original (for drawing clarity)
live = src.resize((dispW,dispH), Image.NEAREST).copy()
dl = ImageDraw.Draw(live, "RGBA")

# Draw existing polygon + nodes
if ss.poly:
    ptsc=[o2c(p) for p in ss.poly]
    if len(ss.poly)>=2: dl.line(ptsc, fill=(0,255,255,255), width=2)
    if len(ss.poly)>=3 and ss.poly and ss.poly[0]==ss.poly[-1]:
        dl.polygon(ptsc, outline=(0,255,255,255), fill=(0,255,255,35))
    for (x,y) in ptsc:
        dl.ellipse([x-4,y-4,x+4,y+4], fill=(0,255,255,220))

# Draw joint lines (both segments), extended full image width
joint_colors = {"proximal": (0,255,255,255), "distal": (221,0,221,255)}
for seg in ("proximal","distal"):
    jl = ss.joint_lines.get(seg)
    if jl:
        j0d, j1d = o2c(jl["p0"]), o2c(jl["p1"])
        a,b = extend_inf_line_through_image(j0d, j1d, dispW, dispH)
        dl.line([a,b], fill=joint_colors[seg], width=2)
        for q in [j0d, j1d]:
            dl.ellipse([q[0]-4,q[1]-4,q[0]+4,q[1]+4], fill=joint_colors[seg])

# Draw axes + endpoints
for ax in ss.axes:
    col = (66,133,244,255) if ax["segment"]=="proximal" else (255,105,180,255)  # blue for prox, pink for distal
    p0d,p1d = o2c(ax["p0"]), o2c(ax["p1"])
    dl.line([p0d,p1d], fill=col, width=3)
    dl.ellipse([p0d[0]-4,p0d[1]-4,p0d[0]+4,p0d[1]+4], fill=(255,255,255,220))
    dl.ellipse([p1d[0]-4,p1d[1]-4,p1d[0]+4,p1d[1]+4], fill=(255,255,255,220))

# Pending node markers for current placement (so clicks feel immediate)
if ss.tool=="Joint line" and ss.placing_joint is not None:
    p0d = o2c(ss.placing_joint["p0"])
    dl.ellipse([p0d[0]-5,p0d[1]-5,p0d[0]+5,p0d[1]+5], outline=(255,255,0,255), width=2)
if ss.tool=="Axis" and ss.placing_axis is not None:
    p0d = o2c(ss.placing_axis["p0"])
    dl.ellipse([p0d[0]-5,p0d[1]-5,p0d[0]+5,p0d[1]+5], outline=(255,255,0,255), width=2)

# Hinge marker
if ss.hinge:
    hx,hy=o2c(ss.hinge)
    dl.ellipse([hx-7,hy-7,hx+7,hy+7], outline=(255,165,0,255), width=3)
    dl.line([(hx-12,hy),(hx+12,hy)], fill=(255,165,0,255), width=1)
    dl.line([(hx,hy-12),(hx,hy+12)], fill=(255,165,0,255), width=1)

# Show live & capture click (RGB to avoid JPEG alpha issue)
left_image_slot = left.empty()
left_image_slot.image(live, use_column_width=False, width=dispW)
click = streamlit_image_coordinates(live.convert("RGB"), width=dispW, key=f"click-{ss.click_nonce}")
xo=yo=None
if click and "x" in click and "y" in click:
    xo,yo = c2o((float(click["x"]), float(click["y"])))

# --------------- CLICK HANDLERS (single-click, no forced rerun) ---------------
def toast(msg): st.toast(msg)

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
    # small toolbar inline to avoid sidebar jumping
    cA,cB,cC,cD = left.columns([1,1,1,1])
    if cA.button("Undo"):
        if ss.poly: ss.poly.pop(); toast("Polygon point removed")
    if cB.button("Clear"):
        ss.poly.clear(); toast("Polygon cleared")
    if cC.button("Close"):
        if len(ss.poly)>=3:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
    cD.markdown("&nbsp;")
    if xo is not None:
        if len(ss.poly)>=2 and math.hypot(xo-ss.poly[0][0], yo-ss.poly[0][1]) < 12/scale:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
        else:
            ss.poly.append((xo,yo)); toast("Point added")

elif ss.tool == "Hinge":
    if xo is not None:
        ss.hinge = (xo,yo); toast("Hinge set")

# --------------- PREVIEW / SIMULATION (right) ---------------
# Base: overlay-style simulation result if in Simulate tool (or using quick sliders)
move_now = (ss.tool=="Simulate")
if move_now and len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1] and ss.hinge:
    preview_img = apply_osteotomy_overlay(src, ss.poly, ss.hinge, ss.dx, ss.dy, ss.theta, ss.segment)
else:
    preview_img = src

disp = preview_img.resize((dispW,dispH), Image.NEAREST).copy()
dr = ImageDraw.Draw(disp, "RGBA")

# polygon on preview
if ss.poly:
    ptsc=[o2c(p) for p in ss.poly]
    if len(ss.poly)>=2: dr.line(ptsc, fill=(0,255,255,255), width=2)
    if len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1]:
        dr.polygon(ptsc, outline=(0,255,255,255), fill=(0,255,255,35))
    for (x,y) in ptsc:
        dr.ellipse([x-4,y-4,x+4,y+4], fill=(0,255,255,220))

# joint lines on preview
for seg in ("proximal","distal"):
    jl = ss.joint_lines.get(seg)
    if jl:
        j0d, j1d = o2c(jl["p0"]), o2c(jl["p1"])
        a,b = extend_inf_line_through_image(j0d, j1d, dispW, dispH)
        dr.line([a,b], fill=joint_colors[seg], width=2)
        for q in [j0d, j1d]:
            dr.ellipse([q[0]-4,q[1]-4,q[0]+4,q[1]+4], fill=joint_colors[seg])

# axes on preview
angles_sidebar = []
for ax in ss.axes:
    p0, p1 = ax["p0"], ax["p1"]

    # if simulating, transform the axis endpoints that belong to the moving segment
    if move_now and len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1] and ss.hinge:
        if ax["segment"] == ss.segment:
            p0, p1 = transform_points_screen([p0,p1], ss.dx, ss.dy, ss.theta, ss.hinge)
            p0, p1 = p0[0], p1[0]  # unpack list of tuples

    col = (66,133,244,255) if ax["segment"]=="proximal" else (255,105,180,255)
    dr.line([o2c(p0), o2c(p1)], fill=col, width=3)
    dr.ellipse([o2c(p0)[0]-4,o2c(p0)[1]-4,o2c(p0)[0]+4,o2c(p0)[1]+4], fill=(255,255,255,220))
    dr.ellipse([o2c(p1)[0]-4,o2c(p1)[1]-4,o2c(p1)[0]+4,o2c(p1)[1]+4], fill=(255,255,255,220))

    # angle vs its corresponding joint line (if present)
    jl = ss.joint_lines.get(ax["segment"])
    if jl:
        j0d, j1d = o2c(jl["p0"]), o2c(jl["p1"])
        acute, obtuse = angle_between_lines_deg(j0d, j1d, o2c(p0), o2c(p1))
        if acute is not None:
            mid = ((o2c(p0)[0]+o2c(p1)[0])/2, (o2c(p0)[1]+o2c(p1)[1])/2)
            draw_label(disp, (mid[0]+6, mid[1]+6), f"{acute:.1f}°/{obtuse:.1f}°")
            angles_sidebar.append((ax["segment"], acute, obtuse))

# hinge on preview
if ss.hinge:
    hx,hy=o2c(ss.hinge)
    dr.ellipse([hx-7,hy-7,hx+7,hy+7], outline=(255,165,0,255), width=3)
    dr.line([(hx-12,hy),(hx+12,hy)], fill=(255,165,0,255), width=1)
    dr.line([(hx,hy-12),(hx,hy+12)], fill=(255,165,0,255), width=1)

right.image(disp, use_column_width=False, width=dispW)

# angles list in sidebar
if angles_sidebar:
    st.sidebar.markdown("### Axis ∠ vs joint line")
    for i,(seg,a,o) in enumerate(angles_sidebar,1):
        st.sidebar.write(f"{seg.capitalize()} axis {i}: **{a:.1f}° / {o:.1f}°**")
