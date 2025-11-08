# streamlit_app.py
import io, math, hashlib
from typing import List, Tuple, Optional
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – snappy single image", layout="wide")

Pt = Tuple[float, float]
Line = List[Pt]

# ---------- tiny utils ----------
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def load_rgba(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(img).convert("RGBA")

def polygon_mask(size: Tuple[int,int], pts: List[Pt]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3: ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def centroid(pts: List[Pt]) -> Optional[Pt]:
    if len(pts) < 3: return None
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    a=cx=cy=0.0
    for i in range(len(pts)):
        j=(i+1)%len(pts); cross=x[i]*y[j]-x[j]*y[i]
        a += cross; cx += (x[i]+x[j])*cross; cy += (y[i]+y[j])*cross
    a*=0.5
    return None if abs(a)<1e-9 else (cx/(6*a), cy/(6*a))

def apply_affine_fragment(moving: Image.Image, dx: float, dy: float, rot_deg: float, center_xy: Pt) -> Image.Image:
    # Pillow rotation is CCW in screen (y-down) coordinates
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", moving.size, (0,0,0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def rotate_point(p: Pt, c: Pt, deg: float) -> Pt:
    ang=math.radians(deg); cos,sin=math.cos(ang),math.sin(ang)
    x,y=p[0]-c[0], p[1]-c[1]
    xr =  x*cos + y*sin
    yr = -x*sin + y*cos
    return (c[0]+xr, c[1]+yr)

def transform_line(line: Line, c: Pt, dx: float, dy: float, theta: float) -> Line:
    p0=rotate_point(line[0],c,theta); p1=rotate_point(line[1],c,theta)
    return [(p0[0]+dx,p0[1]+dy),(p1[0]+dx,p1[1]+dy)]

def safe_width_slider(default_hint: int, uploaded_img: Optional[Image.Image]) -> int:
    min_w = 200
    max_w = max(min_w+1, min(1800, uploaded_img.size[0] if uploaded_img else 1200))
    default = max(min_w+1, min(default_hint, max_w))
    return st.sidebar.slider("Preview width", min_value=min_w, max_value=max_w, value=default, step=50)

# vector helpers
def _vec(p0: Pt, p1: Pt) -> Pt: return (p1[0]-p0[0], p1[1]-p0[1])
def _norm(v: Pt) -> float: return (v[0]*v[0]+v[1]*v[1])**0.5
def _unit(v: Pt) -> Pt:
    n=_norm(v);  return (0.0,0.0) if n==0 else (v[0]/n, v[1]/n)
def _perp(v: Pt) -> Pt: return (-v[1], v[0])

def angle_between_lines(l1: Line, l2: Line) -> Optional[Tuple[float,float]]:
    """(small_deg, large_deg) in 0..180"""
    if len(l1)!=2 or len(l2)!=2: return None
    v1=_unit(_vec(l1[0],l1[1])); v2=_unit(_vec(l2[0],l2[1]))
    if _norm(v1)==0 or _norm(v2)==0: return None
    dot=max(-1.0,min(1.0, v1[0]*v2[0]+v1[1]*v2[1]))
    small = math.degrees(math.acos(dot))
    return (small, 180.0-small)

def line_intersection(l1: Line, l2: Line) -> Optional[Pt]:
    if len(l1)!=2 or len(l2)!=2: return None
    x1,y1=l1[0]; x2,y2=l1[1]; x3,y3=l2[0]; x4,y4=l2[1]
    den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(den)<1e-9: return None
    px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
    py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
    return (px,py)

def angle_bisectors(l1: Line, l2: Line) -> Optional[Tuple[Pt, Pt]]:
    """
    Return unit vectors for the small-angle bisector and the opposite (large-angle) bisector
    in screen (y-down) coords, oriented from the intersection point.
    """
    if len(l1)!=2 or len(l2)!=2: return None
    v1=_unit(_vec(l1[0],l1[1])); v2=_unit(_vec(l2[0],l2[1]))
    if _norm(v1)==0 or _norm(v2)==0: return None
    # small-angle bisector ~ normalize(v1 + v2); if near zero (opposite), try v1 - v2
    cand = (v1[0]+v2[0], v1[1]+v2[1])
    if _norm(cand) < 1e-6:
        cand = (v1[0]-v2[0], v1[1]-v2[1])
    b_small = _unit(cand)
    b_large = (-b_small[0], -b_small[1])
    return (b_small, b_large)

def bubble(d: ImageDraw.ImageDraw, anchor: Pt, text: str):
    pad=4
    tw=max(40,int(8*max(1,len(text))*0.55)); th=16
    bx0,by0=anchor[0]+8,anchor[1]-8; bx1,by1=bx0+tw+pad*2,by0+th
    d.rectangle([bx0,by0,bx1,by1], fill=(0,0,0,170))
    d.text((bx0+pad, by0+2), text, fill=(255,255,255,230))

def map_sliders_to_screen(dx_slider: float, dy_slider: float, prox_axis: Line) -> Tuple[float,float]:
    """ΔY moves ∥ proximal axis; ΔX moves ⟂ proximal axis (CCW 90°)."""
    if len(prox_axis)==2:
        v=_unit(_vec(prox_axis[0],prox_axis[1]))      # parallel
        n=_unit(_perp(v))                             # perpendicular
        return (dx_slider*n[0] + dy_slider*v[0], dx_slider*n[1] + dy_slider*v[1])
    return (dx_slider, dy_slider)

# ---------- cache resized preview ----------
@st.cache_data(show_spinner=False)
def _resize_for_display(img_bytes_hash: str, target_w: int, raw_bytes: bytes) -> Image.Image:
    pil = load_rgba(raw_bytes)
    W,H = pil.size
    scale = min(target_w/float(W), 1.0)
    return pil.resize((int(round(W*scale)), int(round(H*scale))), Image.NEAREST)

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
st.sidebar.title("Osteotomy visualizer")
st.sidebar.caption("by **Nath Adulkasem, MD, PhD**")  # <-- credit line

st.sidebar.header("Load image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool = st.sidebar.radio(
    "Tool",
    ["Osteotomy","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"],
    index=["Osteotomy","Prox axis","Dist axis","Prox joint","Dist joint","HINGE","CORA"].index(ss.tool),
)

st.sidebar.markdown("**Delete a single item**")
del_choice = st.sidebar.selectbox("(select to clear one)",
                                  ["(none)","Prox axis","Dist axis","Prox joint","Dist joint"])
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

ss.move_segment = st.sidebar.radio("Move which part?", ["distal","proximal"],
                                   horizontal=True, index=0 if ss.move_segment=="distal" else 1)

probe_img = load_rgba(up.getvalue()) if up else None
ss.dispw = safe_width_slider(ss.dispw, probe_img)

st.sidebar.markdown("---")
# tightened ranges per your request
ss.dx    = st.sidebar.slider("ΔX (⟂ prox axis)  px", -200, 200, ss.dx, 1)
ss.dy    = st.sidebar.slider("ΔY (∥ prox axis) px", -200, 200, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -60.0, 60.0, float(ss.theta), 0.2)

# ---------------- main ----------------
if not up:
    st.info("Upload an image to begin.")
    st.stop()

raw_bytes = up.getvalue()
disp = _resize_for_display(_hash_bytes(raw_bytes), ss.dispw, raw_bytes)

dx_screen, dy_screen = map_sliders_to_screen(ss.dx, ss.dy, ss.prox_axis)
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
    base.alpha_composite(fixed); base.alpha_composite(moved)
    composite = base

def overlay_img() -> Image.Image:
    img = composite.convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    # osteotomy polygon
    if ss.poly:
        if len(ss.poly) >= 2: d.line(ss.poly, fill=(0,255,255,255), width=2)
        if ss.poly_closed and len(ss.poly) >= 3:
            d.line([ss.poly[-1], ss.poly[0]], fill=(0,255,255,255), width=2)
        for p in ss.poly:
            d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=(0,255,255,200))

    # choose which lines to draw (moving segment follows image)
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
        if len(line)>=1:
            p0=line[0]; d.ellipse([p0[0]-4,p0[1]-4,p0[0]+4,p0[1]+4], fill=col)
        if len(line)==2:
            d.line(line, fill=col, width=3)
            for p in line: d.ellipse([p[0]-4,p[1]-4,p[0]+4,p[1]+4], fill=col)
            mid=((line[0][0]+line[1][0])/2.0,(line[0][1]+line[1][1])/2.0)
            bubble(d, mid, label)

    _draw_line(prox_axis,(66,133,244,255),"prox axis")
    _draw_line(dist_axis,(221,0,221,255),"dist axis")
    _draw_line(prox_joint,(255,215,0,220),"prox joint")
    _draw_line(dist_joint,(255,215,0,220),"dist joint")

    # angle readouts as two *separate* bubbles placed along bisectors
    def _two_bubbles(name: str, l1: Line, l2: Line):
        if len(l1)!=2 or len(l2)!=2: return
        inter = line_intersection(l1, l2)
        if not inter:
            # fallback to midpoint between midpoints
            m1=((l1[0][0]+l1[1][0])/2.0,(l1[0][1]+l1[1][1])/2.0)
            m2=((l2[0][0]+l2[1][0])/2.0,(l2[0][1]+l2[1][1])/2.0)
            inter=((m1[0]+m2[0])/2.0,(m1[1]+m2[1])/2.0)
        ab = angle_between_lines(l1, l2)
        bis = angle_bisectors(l1, l2)
        if not ab or not bis: return
        small, large = ab
        b_small, b_large = bis
        # place α and β along their respective bisectors
        r1, r2 = 28, 52  # offsets
        p_small = (inter[0] + b_small[0]*r1, inter[1] + b_small[1]*r1)
        p_large = (inter[0] + b_large[0]*r2, inter[1] + b_large[1]*r2)
        bubble(d, p_small, f"{name}  α={small:.1f}°")
        bubble(d, p_large, f"{name}  β={large:.1f}°")

    # 1) prox joint vs prox axis
    if len(prox_axis)==2 and len(prox_joint)==2: _two_bubbles("prox joint↔axis", prox_joint, prox_axis)
    # 2) dist joint vs dist axis
    if len(dist_axis)==2 and len(dist_joint)==2: _two_bubbles("dist joint↔axis", dist_joint, dist_axis)
    # 3) prox axis vs dist axis
    if len(prox_axis)==2 and len(dist_axis)==2:  _two_bubbles("prox axis↔dist axis", prox_axis, dist_axis)

    if ss.hinge:
        x,y=ss.hinge
        d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)
    if ss.cora:
        x,y=ss.cora; d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0,255), width=2)
    return img.convert("RGB")

overlay_rgb = overlay_img()
click = streamlit_image_coordinates(overlay_rgb, width=overlay_rgb.size[0], key="click")

if click and "x" in click and "y" in click:
    p = (float(click["x"]), float(click["y"]))
    t = ss.tool
    if t == "Osteotomy":
        if not ss.poly_closed:
            if len(ss.poly) >= 3:
                x0,y0=ss.poly[0]
                if (p[0]-x0)**2 + (p[1]-y0)**2 <= 10**2:
                    ss.poly_closed=True
                else:
                    ss.poly.append(p)
            else:
                ss.poly.append(p)
    elif t == "Prox axis":
        if len(ss.prox_axis)<1: ss.prox_axis=[p]
        elif len(ss.prox_axis)==1: ss.prox_axis.append(p)
        else: ss.prox_axis=[p]
    elif t == "Dist axis":
        if len(ss.dist_axis)<1: ss.dist_axis=[p]
        elif len(ss.dist_axis)==1: ss.dist_axis.append(p)
        else: ss.dist_axis=[p]
    elif t == "Prox joint":
        if len(ss.prox_joint)<1: ss.prox_joint=[p]
        elif len(ss.prox_joint)==1: ss.prox_joint.append(p)
        else: ss.prox_joint=[p]
    elif t == "Dist joint":
        if len(ss.dist_joint)<1: ss.dist_joint=[p]
        elif len(ss.dist_joint)==1: ss.dist_joint.append(p)
        else: ss.dist_joint=[p]
    elif t == "HINGE":
        ss.hinge=p
    elif t == "CORA":
        ss.cora=p
    st.rerun()

with st.expander("Status / help", expanded=False):
    st.write(f"**Tool**: {ss.tool}  |  Osteotomy closed: {ss.poly_closed}")
    st.write("Click once to place a node; the line appears after the 2nd click. "
             "Click near the first node to close the osteotomy.  ΔY slides ∥ prox axis; ΔX slides ⟂ prox axis. "
             "Angle labels show α (small) and β (large) positioned along the corresponding bisectors.")
