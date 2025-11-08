# app.py
import io, base64
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – live canvas over real image", layout="wide")

# ---------------- helpers ----------------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def parse_line(obj):
    return [(float(obj.get("x1",0)), float(obj.get("y1",0))),
            (float(obj.get("x2",0)), float(obj.get("y2",0)))]

def parse_poly_points(obj):
    # Prefer Fabric 'points'
    if obj.get("type") in ("polyline","polygon") and isinstance(obj.get("points"), list):
        left = float(obj.get("left",0)); top = float(obj.get("top",0))
        sx = float(obj.get("scaleX",1)); sy = float(obj.get("scaleY",1))
        return [(left + sx*float(p.get("x",0)), top + sy*float(p.get("y",0))) for p in obj["points"]]
    # Fallback 'path'
    if "path" in obj and isinstance(obj["path"], list):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M","L"):
                pts.append((float(cmd[1]), float(cmd[2])))
        return pts
    return []

def sqdist(a,b): dx=a[0]-b[0]; dy=a[1]-b[1]; return dx*dx + dy*dy

def pil_to_dataurl(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# A robust canvas caller: tries (A) locked image via initial_drawing (best),
# then (B) PIL background, (C) NumPy background, (D) no background.
def safe_canvas(background_pil, *, drawing_mode, stroke_color, stroke_width, key, width, height, update_streamlit=True):
    from streamlit_drawable_canvas import st_canvas

    # A) Locked Fabric image underlay (works everywhere)
    init = {
        "version": "5.2.4",
        "objects": [{
            "type": "image",
            "src": pil_to_dataurl(background_pil),
            "left": 0, "top": 0,
            "width": width, "height": height,
            "scaleX": 1, "scaleY": 1,
            "selectable": False, "evented": False,
            "hasControls": False, "hasBorders": False,
        }]
    }
    try:
        return st_canvas(
            initial_drawing=init,
            background_color=None,                 # important: don't overwrite the image
            fill_color="rgba(0,0,0,0)",
            drawing_mode=drawing_mode,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            update_streamlit=update_streamlit,
            width=width, height=height,
            key=key,
            display_toolbar=True,
            display_ratio=1.0,                     # fixes Hi-DPI drift if supported
        )
    except TypeError:
        pass
    except Exception:
        pass

    # B) PIL background
    try:
        return st_canvas(
            background_image=background_pil,
            background_color="#ffffff",
            fill_color="rgba(0,0,0,0)",
            drawing_mode=drawing_mode,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            update_streamlit=update_streamlit,
            width=width, height=height,
            key=key,
            display_toolbar=True,
            display_ratio=1.0,
        )
    except Exception:
        pass

    # C) NumPy background
    try:
        return st_canvas(
            background_image=np.array(background_pil.convert("RGB")),
            background_color="#ffffff",
            fill_color="rgba(0,0,0,0)",
            drawing_mode=drawing_mode,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            update_streamlit=update_streamlit,
            width=width, height=height,
            key=key,
            display_toolbar=True,
        )
    except Exception:
        pass

    # D) Last resort: no background (you will still see the preview)
    return st_canvas(
        background_color="#ffffff",
        fill_color="rgba(0,0,0,0)",
        drawing_mode=drawing_mode,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        update_streamlit=update_streamlit,
        width=width, height=height,
        key=key,
        display_toolbar=True,
    )

# ---------------- state ----------------
ss = st.session_state
for k,v in dict(tool="Polygon", preview_w=1100, snap_px=12,
                poly=[], prox=[], dist=[], cora=None, hinge=None,
                poly_open=True, nonce=0).items():
    ss.setdefault(k, v)

# ---------------- sidebar ----------------
st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])
ss.tool      = st.sidebar.radio("Tool", ["Polygon","Prox line","Dist line","CORA","HINGE"], index=0)
ss.preview_w = st.sidebar.slider("Preview width", 600, 1800, ss.preview_w, 50)
ss.snap_px   = st.sidebar.slider("Polygon snap distance (px)", 4, 20, int(ss.snap_px), 1)
c1,c2,c3,c4 = st.sidebar.columns(4)
if c1.button("Reset poly"):  ss.poly.clear(); ss.poly_open=True; ss.nonce += 1
if c2.button("Reset lines"): ss.prox.clear(); ss.dist.clear(); ss.nonce += 1
if c3.button("Clear points"): ss.cora=None; ss.hinge=None; ss.nonce += 1
if c4.button("Resync"):      ss.nonce += 1

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# ---------------- image + unified sizing ----------------
img = load_rgba(up.getvalue())
origW, origH = img.size
cw = min(ss.preview_w, origW)
scale = cw / float(origW)          # one scale used everywhere
ch = int(round(origH * scale))

def c2o(pt): return (pt[0]/scale, pt[1]/scale)   # canvas->orig
def o2c(pt): return (pt[0]*scale, pt[1]*scale)   # orig->canvas

disp_img = img.resize((cw, ch), Image.NEAREST)

# ---------------- layout ----------------
left, right = st.columns([1.05, 1])
left.subheader("Live drawing (real image background)")
right.subheader("Preview (persisted). Click to set CORA / HINGE")

# ---------------- canvas ----------------
tool = ss.tool
if tool == "Polygon":
    drawing_mode = "polyline" if ss.poly_open else "transform"  # live + snap close
    stroke_color = "#00FFFF"; stroke_w = 2
elif tool == "Prox line":
    drawing_mode = "line"; stroke_color = "#4285F4"; stroke_w = 3
elif tool == "Dist line":
    drawing_mode = "line"; stroke_color = "#DD00DD"; stroke_w = 3
else:
    drawing_mode = "transform"; stroke_color = "#00FFFF"; stroke_w = 2

with left:
    result = safe_canvas(
        background_pil=disp_img,
        drawing_mode=drawing_mode,
        stroke_color=stroke_color,
        stroke_width=stroke_w,
        key=f"canvas-{ss.nonce}-{tool}",
        width=cw, height=ch,
        update_streamlit=True,
    )

# -------- capture shapes from canvas (persist in ORIGINAL pixels) --------
if result.json_data and tool in ("Polygon","Prox line","Dist line"):
    objs = result.json_data.get("objects", [])
    for obj in reversed(objs):
        typ = obj.get("type","")
        if typ == "image":
            continue
        if tool in ("Prox line","Dist line") and typ == "line":
            seg = parse_line(obj)
            if len(seg) == 2:
                seg_o = [c2o(seg[0]), c2o(seg[1])]
                if tool == "Prox line": ss.prox = seg_o
                else: ss.dist = seg_o
            break
        if tool == "Polygon" and ss.poly_open and typ in ("polyline","polygon","path"):
            pts = parse_poly_points(obj)
            if len(pts) >= 2:
                p0, pend = pts[0], pts[-1]
                if sqdist(p0, pend) <= (ss.snap_px*ss.snap_px) and len(pts) >= 3:
                    pts[-1] = p0
                    ss.poly = [c2o(p) for p in pts]
                    ss.poly_open = False
                    ss.nonce += 1
                    st.experimental_rerun()
                else:
                    ss.poly = [c2o(p) for p in pts]
            break

# ---------------- preview (same cw×ch -> perfect alignment) ----------------
preview = disp_img.copy()
d = ImageDraw.Draw(preview, "RGBA")
if len(ss.poly) >= 3:
    d.polygon([o2c(p) for p in ss.poly], outline=(0,255,255,255), fill=(0,255,255,40))
if len(ss.prox) == 2:
    d.line([o2c(ss.prox[0]), o2c(ss.prox[1])], fill=(66,133,244,255), width=3)
if len(ss.dist) == 2:
    d.line([o2c(ss.dist[0]), o2c(ss.dist[1])], fill=(221,0,221,255), width=3)
if ss.cora:
    x,y = o2c(ss.cora); d.ellipse([x-5,y-5,x+5,y+5], outline=(0,200,0,255), width=2)
if ss.hinge:
    x,y = o2c(ss.hinge)
    d.ellipse([x-6,y-6,x+6,y+6], outline=(255,165,0,255), width=3)
    d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
    d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)

with right:
    if tool in ("CORA","HINGE"):
        st.caption("Click on the image to set the point.")
        click = streamlit_image_coordinates(preview, width=cw, key=f"clicks-{ss.nonce}")
        if click and "x" in click and "y" in click:
            pt_o = (float(click["x"])/scale, float(click["y"])/scale)
            if tool == "CORA": ss.cora = pt_o
            else: ss.hinge = pt_o
            ss.nonce += 1
            st.experimental_rerun()
    else:
        st.image(preview, width=cw)
