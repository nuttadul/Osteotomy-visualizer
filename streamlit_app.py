# app.py — SAFE MODE (no third-party canvas required) + optional experimental canvas
import io, base64
from typing import List, Tuple
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy – Safe Mode", layout="wide")

# ---------------- helpers ----------------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def draw_overlay(base_rgba: Image.Image, state) -> Image.Image:
    out = base_rgba.copy()
    d = ImageDraw.Draw(out, "RGBA")
    # polygon
    if len(state["poly"]) >= 2:
        pts = [o2c(p, state["scale"]) for p in state["poly"]]
        if len(pts) >= 3 and not state["poly_open"]:
            d.polygon(pts, outline=(0,255,255,255), fill=(0,255,255,40))
        else:
            d.line(pts, fill=(0,255,255,255), width=2)
    # lines
    if len(state["prox"]) == 2:
        d.line([o2c(state["prox"][0], state["scale"]), o2c(state["prox"][1], state["scale"])],
               fill=(66,133,244,255), width=3)
    if len(state["dist"]) == 2:
        d.line([o2c(state["dist"][0], state["scale"]), o2c(state["dist"][1], state["scale"])],
               fill=(221,0,221,255), width=3)
    # points
    if state["cora"]:
        x,y = o2c(state["cora"], state["scale"]); d.ellipse([x-5,y-5,x+5,y+5], outline=(0,200,0,255), width=2)
    if state["hinge"]:
        x,y = o2c(state["hinge"], state["scale"])
        d.ellipse([x-6,y-6,x+6,y+6], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)
    return out

def c2o(pt, scale):  # canvas(pixel) -> original(pixel)
    return (pt[0] / scale, pt[1] / scale)

def o2c(pt, scale):  # original(pixel) -> canvas(pixel)
    return (pt[0] * scale, pt[1] * scale)

def close_enough(a, b, px):  # squared distance check
    dx = a[0]-b[0]; dy = a[1]-b[1]
    return dx*dx + dy*dy <= px*px

# ---------------- state ----------------
ss = st.session_state
defaults = dict(
    tool="Polygon",
    preview_w=1100,
    snap_px=12,
    poly=[], poly_open=True,
    prox=[], dist=[],
    cora=None, hinge=None,
    use_canvas=False,   # experimental (off by default)
    nonce=0
)
for k,v in defaults.items():
    ss.setdefault(k,v)

# ---------------- sidebar ----------------
st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool        = st.sidebar.radio("Tool", ["Polygon","Prox line","Dist line","CORA","HINGE"], index=0)
ss.preview_w   = st.sidebar.slider("Preview width", 600, 1800, ss.preview_w, 50)
ss.snap_px     = st.sidebar.slider("Polygon snap distance (px)", 4, 20, int(ss.snap_px), 1)
ss.use_canvas  = st.sidebar.toggle("Experimental live canvas (may be unstable)", value=ss.use_canvas)

c1,c2,c3,c4,c5 = st.sidebar.columns(5)
if c1.button("Undo"):
    if ss.tool == "Polygon" and ss.poly:
        ss.poly.pop()
    elif ss.tool == "Prox line" and ss.prox:
        ss.prox.pop()
    elif ss.tool == "Dist line" and ss.dist:
        ss.dist.pop()
if c2.button("Finish poly"):
    if len(ss.poly) >= 3:
        ss.poly_open = False
if c3.button("Reset poly"):   ss.poly.clear(); ss.poly_open=True
if c4.button("Reset lines"):  ss.prox.clear(); ss.dist.clear()
if c5.button("Clear points"): ss.cora=None; ss.hinge=None

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# ---------------- sizing ----------------
img = load_rgba(up.getvalue())
origW, origH = img.size
cw = min(ss.preview_w, origW)
scale = cw / float(origW)
ch = int(round(origH * scale))
disp = img.resize((cw, ch), Image.NEAREST)

state = dict(scale=scale, poly=ss.poly, poly_open=ss.poly_open,
             prox=ss.prox, dist=ss.dist, cora=ss.cora, hinge=ss.hinge)

left, right = st.columns([1.05,1])

# ---------------- LEFT: interactive input ----------------
left.subheader("Input")
if ss.use_canvas:
    # optional: try canvas (image always shown as plain background in left panel)
    st.caption("Experimental canvas disabled in Safe Mode build — use the click workflow below.")
    # (Intentionally not calling the 3rd-party canvas here to avoid the crash loop)
else:
    # click-to-place workflow (robust)
    # 1) show current overlay image and capture one click
    live = draw_overlay(disp, state)
    res = left.image(live, use_column_width=False)
    click = streamlit_image_coordinates(live, width=cw, key=f"clicks-{ss.nonce}")

    if click and "x" in click and "y" in click:
        pt_c = (float(click["x"]), float(click["y"]))
        pt_o = c2o(pt_c, scale)

        if ss.tool == "Polygon":
            if ss.poly_open:
                if len(ss.poly) >= 2 and close_enough(o2c(ss.poly[0], scale), pt_c, ss.snap_px):
                    # auto-close
                    ss.poly_open = False
                else:
                    ss.poly.append(pt_o)
        elif ss.tool == "Prox line":
            ss.prox.append(pt_o)
            if len(ss.prox) > 2: ss.prox = ss.prox[-2:]
        elif ss.tool == "Dist line":
            ss.dist.append(pt_o)
            if len(ss.dist) > 2: ss.dist = ss.dist[-2:]
        elif ss.tool == "CORA":
            ss.cora = pt_o
        elif ss.tool == "HINGE":
            ss.hinge = pt_o
        ss.nonce += 1
        st.rerun()

# ---------------- RIGHT: preview (always aligned, always shows background) ----------------
right.subheader("Preview")
final_overlay = draw_overlay(disp, dict(
    scale=scale, poly=ss.poly, poly_open=ss.poly_open,
    prox=ss.prox, dist=ss.dist, cora=ss.cora, hinge=ss.hinge
))
right.image(final_overlay, use_column_width=False)

st.caption("Safe Mode: reliable online. Turn on 'Experimental live canvas' later if you need drag-lines and your host supports it.")
