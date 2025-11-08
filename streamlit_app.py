# app.py — Bone Ninja–style Osteotomy Planner
import io, math, base64
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Bone Ninja-style Osteotomy", layout="wide")

# ---------------- Utilities ----------------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def c2o(pt, scale): return (pt[0]/scale, pt[1]/scale)
def o2c(pt, scale): return (pt[0]*scale, pt[1]*scale)

def vec_angle(v1, v2):
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    deg = math.degrees(abs(a1 - a2))
    return min(deg, 180 - deg if deg > 180 else deg)

def safe_canvas(bg_img, drawing_mode, stroke_color, stroke_width, key, width, height):
    """Always show background image — compatible with all Streamlit versions."""
    import numpy as np
    bg_np = np.array(bg_img.convert("RGB"))  # convert PIL → NumPy array
    
    return st_canvas(
        background_image=bg_np,
        fill_color="rgba(0,0,0,0)",
        background_color=None,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        width=width,
        height=height,
        update_streamlit=True,
        key=key,
        display_toolbar=True,
        drawing_mode=drawing_mode,
    )
  

def parse_line(obj):
    return [(float(obj.get("x1",0)), float(obj.get("y1",0))),
            (float(obj.get("x2",0)), float(obj.get("y2",0)))]

def parse_poly(obj):
    if "points" in obj:
        left=obj.get("left",0); top=obj.get("top",0)
        sx=obj.get("scaleX",1); sy=obj.get("scaleY",1)
        pts=[(left+sx*p["x"], top+sy*p["y"]) for p in obj["points"]]
        return pts
    if "path" in obj:
        pts=[]
        for cmd in obj["path"]:
            if isinstance(cmd,list) and len(cmd)>=3 and cmd[0] in ("M","L"):
                pts.append((cmd[1],cmd[2]))
        return pts
    return []

# ---------------- State ----------------
ss = st.session_state
defaults = dict(
    tool="Joint line",
    poly=[], joint=[], axis=[], prox=[], dist=[],
    cora=None, hinge=None,
    angle_input=81.0,
    poly_open=True,
    move_seg="distal",
    dx=0, dy=0, rot=0,
    preview_w=1100,
    nonce=0
)
for k,v in defaults.items(): ss.setdefault(k,v)

# ---------------- Sidebar ----------------
st.sidebar.header("Upload X-ray")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
if not up:
    st.info("Upload an X-ray to begin.")
    st.stop()

ss.tool = st.sidebar.radio("Tool", 
    ["Joint line","Axis line (auto)","Prox line","Dist line","Polygon cut","CORA","HINGE"], index=0)
ss.angle_input = st.sidebar.number_input("Joint orientation angle (°)", 40.0, 140.0, ss.angle_input, 0.5)
ss.move_seg = st.sidebar.radio("Move segment", ["distal","proximal"], index=(0 if ss.move_seg=="distal" else 1))
ss.dx   = st.sidebar.slider("ΔX (px)", -400, 400, ss.dx, 1)
ss.dy   = st.sidebar.slider("ΔY (px)", -400, 400, ss.dy, 1)
ss.rot  = st.sidebar.slider("Rotate (°)", -90, 90, ss.rot, 1)
if st.sidebar.button("Reset move"): ss.dx=ss.dy=ss.rot=0
st.sidebar.divider()
if st.sidebar.button("Reset all"):
    for k in ("poly","joint","axis","prox","dist"): ss[k]=[]
    ss.cora=ss.hinge=None; ss.nonce+=1; st.experimental_rerun()

# ---------------- Image sizing ----------------
img = load_rgba(up.getvalue())
W,H = img.size
scale = min(ss.preview_w/W, 1.0)
cw,ch = int(W*scale), int(H*scale)
disp_img = img.resize((cw,ch), Image.NEAREST)

# ---------------- Layout ----------------
left,right = st.columns([1.1,1])
left.subheader("Live Drawing")
right.subheader("Preview / Simulation")

# ---------------- Drawing ----------------
tool=ss.tool
if tool in ("Joint line","Prox line","Dist line","Polygon cut"):
    mode="line" if tool!="Polygon cut" else "polyline"
else:
    mode="transform"
col="#00FFFF" if tool=="Polygon cut" else "#4285F4"
result = safe_canvas(
    disp_img, mode, col, 3,
    key=f"canvas-{ss.nonce}-{tool}", width=cw, height=ch
)

if result.json_data:
    objs=result.json_data.get("objects",[])
    for obj in reversed(objs):
        t=obj.get("type","")
        if t=="image": continue
        if tool=="Joint line" and t=="line":
            ss.joint=[c2o(p,scale) for p in parse_line(obj)]
            # auto axis creation
            if len(ss.joint)==2:
                (x1,y1),(x2,y2)=ss.joint
                vx,vy=x2-x1,y2-y1
                ang=math.atan2(vy,vx)-math.radians(90-ss.angle_input)
                L=200
                ax=(x2+L*math.cos(ang), y2+L*math.sin(ang))
                ss.axis=[(x2,y2),ax]
            break
        if tool=="Polygon cut" and t in ("polyline","path"):
            pts=parse_poly(obj)
            if len(pts)>=3:
                # auto-close
                if ((pts[0][0]-pts[-1][0])**2+(pts[0][1]-pts[-1][1])**2)**0.5<12:
                    pts[-1]=pts[0]
                ss.poly=[c2o(p,scale) for p in pts]
            break
        if tool=="Prox line" and t=="line": ss.prox=[c2o(p,scale) for p in parse_line(obj)]; break
        if tool=="Dist line" and t=="line": ss.dist=[c2o(p,scale) for p in parse_line(obj)]; break

# ---------------- CORA / HINGE click ----------------
if tool in ("CORA","HINGE"):
    res = streamlit_image_coordinates(disp_img, width=cw, key=f"clicks-{ss.nonce}")
    if res and "x" in res:
        pt=(res["x"]/scale,res["y"]/scale)
        if tool=="CORA": ss.cora=pt
        else: ss.hinge=pt
        ss.nonce+=1; st.experimental_rerun()

# ---------------- Simulation ----------------
preview = disp_img.copy()
draw = ImageDraw.Draw(preview,"RGBA")

def draw_line(l,color):
    if len(l)==2: draw.line([o2c(l[0],scale),o2c(l[1],scale)],fill=color,width=3)
draw_line(ss.joint,(0,200,200,255))
draw_line(ss.axis,(255,200,0,255))
draw_line(ss.prox,(66,133,244,255))
draw_line(ss.dist,(221,0,221,255))
if ss.cora:
    x,y=o2c(ss.cora,scale); draw.ellipse([x-5,y-5,x+5,y+5],outline=(0,255,0,255),width=2)
if ss.hinge:
    x,y=o2c(ss.hinge,scale)
    draw.ellipse([x-6,y-6,x+6,y+6],outline=(255,165,0,255),width=3)
    draw.line([(x-12,y),(x+12,y)],fill=(255,165,0,255),width=1)
    draw.line([(x,y-12),(x,y+12)],fill=(255,165,0,255),width=1)

# polygon cut & movement
if len(ss.poly)>=3:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(ss.poly, fill=255)
    inv = ImageOps.invert(mask)
    prox_img=Image.new("RGBA",img.size,(0,0,0,0));prox_img.paste(img,(0,0),inv)
    dist_img=Image.new("RGBA",img.size,(0,0,0,0));dist_img.paste(img,(0,0),mask)
    center = ss.hinge or ss.cora or ss.poly[0]
    def transform(im):
        rot=im.rotate(ss.rot, center=center, resample=Image.BICUBIC)
        out=Image.new("RGBA",im.size,(0,0,0,0))
        out.alpha_composite(rot,(int(ss.dx),int(ss.dy)))
        return out
    moving = dist_img if ss.move_seg=="distal" else prox_img
    fixed  = prox_img if ss.move_seg=="distal" else dist_img
    moved = transform(moving)
    merged=Image.alpha_composite(fixed,moved)
    preview=merged.resize((cw,ch),Image.NEAREST)
    draw=ImageDraw.Draw(preview,"RGBA")
    # redraw polygon
    pts_c=[o2c(p,scale) for p in ss.poly]
    draw.polygon(pts_c,outline=(0,255,255,255),fill=(0,255,255,40))

# ---------------- Output ----------------
right.image(preview,width=cw)
if len(ss.joint)==2 and len(ss.axis)==2:
    v1=np.array([ss.joint[1][0]-ss.joint[0][0], ss.joint[1][1]-ss.joint[0][1]])
    v2=np.array([ss.axis[1][0]-ss.axis[0][0], ss.axis[1][1]-ss.axis[0][1]])
    st.caption(f"Measured angle between joint & axis: {vec_angle(v1,v2):.1f}°")
