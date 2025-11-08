# app.py  (click-based, no canvas) — with click nodes + clearer simulate
import io, math, time
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# ---------- setup ----------
st.set_page_config(page_title="Osteotomy Visualizer (click mode)", layout="wide")

# ---------- helpers ----------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def polygon_mask(size: Tuple[int, int], pts: List[Tuple[float, float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def rotate_about(pt: Tuple[float,float], center: Tuple[float,float], deg: float) -> Tuple[float,float]:
    x,y = pt; cx,cy = center
    ang = math.radians(deg); c,s = math.cos(ang), math.sin(ang)
    x0,y0 = x-cx, y-cy
    xr = x0*c + y0*s
    yr = -x0*s + y0*c
    return (xr+cx, yr+cy)

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

def dist2(a,b) -> float: return (a[0]-b[0])**2 + (a[1]-b[1])**2
def line_angle_deg(p0,p1) -> float: return math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))%180

def snap_endpoint_to_angle(p0,p1,target,tol):
    dx,dy = p1[0]-p0[0], p1[1]-p0[1]
    if dx==0 and dy==0: return p1
    cur = line_angle_deg(p0,p1)
    choices=[target%180,(target+180)%180]
    best=min(choices,key=lambda a:min(abs(a-cur),180-abs(a-cur)))
    delta=min(abs(best-cur),180-abs(best-cur))
    if delta>tol: return p1
    L=math.hypot(dx,dy); ang=math.radians(best)
    return (p0[0]+L*math.cos(ang), p0[1]+L*math.sin(ang))

# ---------- state ----------
ss = st.session_state
defaults = dict(
    tool="Polygon", subtool="Add",
    dispw=1100,
    joints=[], axes=[], placing_axis=None,
    poly=[], hinge=None,
    snap_on=False, snap_deg=81.0, snap_tol=3.0,
    click_nonce=0, last_click=None, toast_ok=True,
)
for k,v in defaults.items(): ss.setdefault(k,v)

# ---------- sidebar ----------
st.sidebar.header("Upload image")
file = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
st.sidebar.markdown("### Display")
ss.dispw = st.sidebar.slider("Preview width", 600, 1800, int(ss.dispw), 50)
st.sidebar.markdown("### Angle assist for axes")
ss.snap_on = st.sidebar.checkbox("Constrain to angle", value=bool(ss.snap_on))
ss.snap_deg = float(st.sidebar.number_input("Target (deg)", value=float(ss.snap_deg), step=0.5, format="%.1f"))
ss.snap_tol = float(st.sidebar.slider("Tolerance (deg)", 0.0, 15.0, float(ss.snap_tol), 0.5))
c1,c2,c3 = st.sidebar.columns(3)
if c1.button("Reset poly"): ss.poly.clear()
if c2.button("Reset axes"): ss.axes.clear(); ss.placing_axis=None
if c3.button("Reset joints/hinge"): ss.joints.clear(); ss.hinge=None

# ---------- image guard ----------
if not file:
    st.info("Upload an image to begin.")
    st.stop()
img = load_rgba(file.getvalue())
W,H = img.size
scale = min(ss.dispw/float(W), 1.0)
dispW, dispH = int(round(W*scale)), int(round(H*scale))
def o2c(p): return (p[0]*scale, p[1]*scale)
def c2o(p): return (p[0]/scale, p[1]/scale)

# ---------- toolbar ----------
L,R = st.columns([1.05,1])
tb = L.columns([1.25,1,1])
L.subheader("Live Drawing")
tools = ["Joints","Axes","Polygon","Hinge","Simulate"]
ss.tool = tb[0].radio("Tool", tools, index=tools.index(ss.tool), horizontal=True, label_visibility="collapsed")
if ss.tool in ("Joints","Axes"):
    ss.subtool = tb[1].radio("Action", ["Add","Delete"], index=["Add","Delete"].index(ss.subtool),
                             horizontal=True, label_visibility="collapsed")
else:
    tb[1].markdown("&nbsp;")
R.subheader("Preview / Simulation")

# ---------- build overlay (left) ----------
overlay = img.resize((dispW,dispH), Image.NEAREST).copy()
d = ImageDraw.Draw(overlay, "RGBA")

# polygon / nodes
if ss.poly:
    ptsc=[o2c(p) for p in ss.poly]
    # edges
    if len(ss.poly)>=2:
        d.line(ptsc, fill=(0,255,255,255), width=2)
    # nodes (cyan)
    for x,y in ptsc:
        d.ellipse([x-4,y-4,x+4,y+4], fill=(0,255,255,220), outline=None)

# joints (gold nodes)
for jx,jy in ss.joints:
    x,y=o2c((jx,jy))
    d.ellipse([x-5,y-5,x+5,y+5], outline=(255,215,0,255), fill=(255,215,0,80), width=2)

# axes (blue lines + endpoints)
for ax in ss.axes:
    p0c, p1c = o2c(ax["p0"]), o2c(ax["p1"])
    d.line([p0c,p1c], fill=(66,133,244,255), width=3)
    d.ellipse([p0c[0]-4,p0c[1]-4,p0c[0]+4,p0c[1]+4], fill=(66,133,244,200))
    d.ellipse([p1c[0]-4,p1c[1]-4,p1c[0]+4,p1c[1]+4], fill=(66,133,244,200))

# hinge (orange)
if ss.hinge:
    x,y=o2c(ss.hinge)
    d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
    d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
    d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)

# axis ghost (origin mark)
if ss.tool=="Axes" and ss.placing_axis is not None:
    p0 = ss.placing_axis["p0"]
    cx,cy=o2c(p0)
    d.line([(cx-10,cy),(cx+10,cy)], fill=(180,180,255,200), width=1)
    d.line([(cx,cy-10),(cx,cy+10)], fill=(180,180,255,200), width=1)

# show + capture click
click = streamlit_image_coordinates(overlay.convert("RGB"), width=dispW, key=f"clk-{ss.click_nonce}")
px=py=None
if click and "x" in click and "y" in click:
    px,py=c2o((float(click["x"]), float(click["y"])))

# ---------- click handlers ----------
def toast(msg):
    # small non-blocking toast
    if ss.toast_ok:
        st.toast(msg)

if ss.tool=="Joints":
    if ss.subtool=="Add" and px is not None:
        ss.joints.append((px,py)); toast("Joint added")
    elif ss.subtool=="Delete" and px is not None and ss.joints:
        j=int(np.argmin([dist2((px,py),p) for p in ss.joints]))
        if math.hypot(px-ss.joints[j][0], py-ss.joints[j][1]) < 25/scale:
            ss.joints.pop(j); toast("Joint removed")

elif ss.tool=="Axes":
    if ss.subtool=="Delete" and px is not None and ss.axes:
        mids=[((a["p0"][0]+a["p1"][0])/2, (a["p0"][1]+a["p1"][1])/2) for a in ss.axes]
        k=int(np.argmin([dist2((px,py),m) for m in mids]))
        if math.hypot(px-mids[k][0], py-mids[k][1]) < 30/scale:
            ss.axes.pop(k); toast("Axis removed")
    else:
        if ss.placing_axis is None and px is not None:
            bind=None; p0=(px,py)
            if ss.joints:
                j=int(np.argmin([dist2((px,py),p) for p in ss.joints]))
                if math.hypot(px-ss.joints[j][0], py-ss.joints[j][1]) < 30/scale:
                    bind=j; p0=ss.joints[j]
            ss.placing_axis={"joint":bind,"p0":p0}
            toast("Axis origin set")
        elif ss.placing_axis is not None and px is not None:
            p0=ss.placing_axis["p0"]; end=(px,py)
            if ss.snap_on:
                end=snap_endpoint_to_angle(p0,end, ss.snap_deg, ss.snap_tol)
            ss.axes.append({"joint":ss.placing_axis["joint"],"p0":p0,"p1":end,"label":f"AX{len(ss.axes)+1}"})
            ss.placing_axis=None; toast("Axis added")

elif ss.tool=="Polygon":
    cols=L.columns([1,1,1,1])
    if cols[0].button("Undo last"):
        if ss.poly: ss.poly.pop(); toast("Last polygon point removed")
    if cols[1].button("Clear poly"):
        ss.poly.clear(); toast("Polygon cleared")
    if cols[2].button("Close poly"):
        if len(ss.poly)>=3:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
    cols[3].markdown("&nbsp;")

    if px is not None:
        if len(ss.poly)>=2 and math.hypot(px-ss.poly[0][0], py-ss.poly[0][1]) < 12/scale:
            ss.poly.append(ss.poly[0]); toast("Polygon closed")
        else:
            ss.poly.append((px,py)); toast("Polygon point added")

elif ss.tool=="Hinge":
    if px is not None:
        ss.hinge=(px,py); toast("Hinge set")

# ---------- right: preview / simulate ----------
if ss.tool=="Simulate":
    if not (len(ss.poly)>=3 and ss.poly[0]==ss.poly[-1]):
        R.warning("Close the polygon first (Polygon tool → Close poly or click near first point).")
    elif ss.hinge is None:
        R.warning("Set the hinge (Hinge tool).")
    else:
        seg = R.radio("Move segment", ["distal","proximal"], horizontal=True)
        dx = R.slider("ΔX (px)", -1000, 1000, 0, 1)
        dy = R.slider("ΔY (px)", -1000, 1000, 0, 1)
        theta = R.slider("Rotate (deg)", -180, 180, 0, 1)
        # With dx=dy=theta=0 the result is IDENTICAL to source (nothing moves)
        out = apply_osteotomy(img, ss.poly, ss.hinge, dx, dy, theta, seg)
        R.image(out.resize((dispW,dispH), Image.NEAREST), use_column_width=False)
else:
    # show the same overlay the user drew on
    R.image(overlay, use_column_width=False, width=dispW)
