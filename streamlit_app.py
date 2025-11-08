import io, base64, math
from typing import List, Tuple
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

# ------------- basic setup -------------
st.set_page_config(page_title="Osteotomy – fresh build", layout="wide")

# -------- helpers --------
def load_rgba(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    return ImageOps.exif_transpose(im).convert("RGBA")

def pil_to_dataurl(img: Image.Image) -> str:
    """PIL -> data URL for a Fabric.js image object."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def sqdist(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return dx*dx + dy*dy

def draw_overlay(canvas_img: Image.Image, poly, prox, dist, cora, hinge) -> Image.Image:
    out = canvas_img.copy()
    d = ImageDraw.Draw(out, "RGBA")
    if len(poly) >= 3:
        d.polygon(poly, outline=(0,255,255,255), fill=(0,255,255,40))
    if len(prox) == 2:
        d.line(prox, fill=(66,133,244,255), width=3)
    if len(dist) == 2:
        d.line(dist, fill=(221,0,221,255), width=3)
    if cora:
        x,y=cora; d.ellipse([x-5,y-5,x+5,y+5], outline=(0,200,0,255), width=2)
    if hinge:
        x,y=hinge
        d.ellipse([x-6,y-6,x+6,y+6], outline=(255,165,0,255), width=3)
        d.line([(x-12,y),(x+12,y)], fill=(255,165,0,255), width=1)
        d.line([(x,y-12),(x,y+12)], fill=(255,165,0,255), width=1)
    return out

def parse_line(obj):
    return [(float(obj.get("x1",0)), float(obj.get("y1",0))),
            (float(obj.get("x2",0)), float(obj.get("y2",0)))]

def parse_poly_points(obj):
    # Prefer Fabric 'points'
    if obj.get("type") in ("polyline","polygon") and isinstance(obj.get("points"), list):
        left = float(obj.get("left",0)); top = float(obj.get("top",0))
        sx = float(obj.get("scaleX",1)); sy = float(obj.get("scaleY",1))
        pts = []
        for p in obj["points"]:
            pts.append((left + sx*float(p.get("x",0)), top + sy*float(p.get("y",0))))
        return pts
    # Fallback 'path'
    if "path" in obj and isinstance(obj["path"], list):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M","L"):
                pts.append((float(cmd[1]), float(cmd[2])))
        return pts
    return []

def polygon_mask(size, pts) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def apply_affine(img: Image.Image, dx, dy, rot_deg, center_xy):
    """Rotate (CCW) around center_xy, then translate."""
    rot = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", img.size, (0,0,0,0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

# -------- session state --------
ss = st.session_state
defaults = dict(
    tool="Polygon",
    preview_w=1100,
    snap_px=12,
    poly=[],
    prox=[],
    dist=[],
    cora=None,
    hinge=None,
    poly_open=True,
    move_segment="distal",
    dx=0, dy=0, theta=0,
    nonce=0,
)
for k,v in defaults.items():
    ss.setdefault(k, v)

# -------- sidebar (controls) --------
st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool = st.sidebar.radio("Tool", ["Polygon","Prox line","Dist line","CORA","HINGE"], index=0)
ss.preview_w = st.sidebar.slider("Display width", 600, 1800, int(ss.preview_w), 50)
ss.snap_px   = st.sidebar.slider("Polygon snap distance", 4, 20, int(ss.snap_px), 1)

c1,c2,c3,c4 = st.sidebar.columns(4)
if c1.button("Reset poly"):  ss.poly.clear(); ss.poly_open=True; ss.nonce += 1
if c2.button("Reset lines"): ss.prox.clear(); ss.dist.clear(); ss.nonce += 1
if c3.button("Clear points"): ss.cora=None; ss.hinge=None; ss.nonce += 1
if c4.button("Resync"): ss.nonce += 1

st.sidebar.divider()
ss.move_segment = st.sidebar.radio("Segment to move", ["distal","proximal"], horizontal=True)
ss.dx    = st.sidebar.slider("ΔX (px)", -1000, 1000, int(ss.dx), 1)
ss.dy    = st.sidebar.slider("ΔY (px)", -1000, 1000, int(ss.dy), 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, int(ss.theta), 1)

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# -------- load & scale --------
src_rgba = load_rgba(up.getvalue())
origW, origH = src_rgba.size
scale = min(ss.preview_w/float(origW), 1.0)
cw, ch = int(round(origW*scale)), int(round(origH*scale))
disp_img = src_rgba.resize((cw, ch), Image.NEAREST)

def c2o(pt):  # canvas -> original pixel coords
    return (pt[0]/scale, pt[1]/scale)

def o2c(pt):  # original -> canvas coords
    return (pt[0]*scale, pt[1]*scale)

# -------- layout --------
left, right = st.columns([1.05, 1])
left.subheader("Live Drawing")
right.subheader("Preview / Simulation")

# -------- robust canvas with real background (no background_image arg) --------
bg_dataurl = pil_to_dataurl(disp_img)
initial_drawing = {
    "version": "5.2.4",
    "objects": [{
        "type": "image",
        "left": 0, "top": 0,
        "width": cw, "height": ch,
        "scaleX": 1, "scaleY": 1,
        "src": bg_dataurl,
        "selectable": False, "evented": False,
        "hasControls": False, "hasBorders": False
    }]
}

if ss.tool == "Polygon":
    drawing_mode = "polyline" if ss.poly_open else "transform"
    stroke_color, stroke_w = "#00FFFF", 2
elif ss.tool == "Prox line":
    drawing_mode = "line"; stroke_color, stroke_w = "#4285F4", 3
elif ss.tool == "Dist line":
    drawing_mode = "line"; stroke_color, stroke_w = "#DD00DD", 3
else:
    drawing_mode = "transform"; stroke_color, stroke_w = "#00FFFF", 2

with left:
    result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        background_color=None,
        stroke_color=stroke_color,
        stroke_width=stroke_w,
        width=cw, height=ch,
        update_streamlit=True,
        key=f"canvas-{ss.nonce}-{ss.tool}",
        display_toolbar=True,
        initial_drawing=initial_drawing,
        drawing_mode=drawing_mode,
    )

# -------- capture shapes (persist in ORIGINAL space) --------
if result.json_data and ss.tool in ("Polygon","Prox line","Dist line"):
    objs = result.json_data.get("objects", [])
    for obj in reversed(objs):
        typ = obj.get("type", "")
        if typ == "image":
            continue

        if ss.tool in ("Prox line","Dist line") and typ == "line":
            seg = parse_line(obj)
            if len(seg) == 2:
                seg_o = [c2o(seg[0]), c2o(seg[1])]
                if ss.tool == "Prox line": ss.prox = seg_o
                else: ss.dist = seg_o
            break

        if ss.tool == "Polygon" and ss.poly_open and typ in ("polyline","polygon","path"):
            pts = parse_poly_points(obj)
            if len(pts) >= 2:
                p0, plast = pts[0], pts[-1]
                if sqdist(p0, plast) <= (ss.snap_px*ss.snap_px) and len(pts) >= 3:
                    pts[-1] = p0  # close
                    ss.poly = [c2o(p) for p in pts]
                    ss.poly_open = False
                    ss.nonce += 1
                    st.experimental_rerun()
                else:
                    ss.poly = [c2o(p) for p in pts]
            break

# -------- preview + simple simulation --------
# base preview image (canvas size)
base_preview = disp_img.copy()

# draw persisted shapes ON PREVIEW canvas coords
poly_c  = [o2c(p) for p in ss.poly] if len(ss.poly)>=3 else []
prox_c  = [o2c(p) for p in ss.prox] if len(ss.prox)==2 else []
dist_c  = [o2c(p) for p in ss.dist] if len(ss.dist)==2 else []
cora_c  = o2c(ss.cora) if ss.cora else None
hinge_c = o2c(ss.hinge) if ss.hinge else None

# click-to-set CORA/HINGE on the preview
with right:
    preview_img = draw_overlay(base_preview, poly_c, prox_c, dist_c, cora_c, hinge_c)

    if ss.tool in ("CORA","HINGE"):
        st.caption("Click to set point.")
        click = streamlit_image_coordinates(preview_img, width=cw, key=f"clicks-{ss.nonce}")
        if click and "x" in click and "y" in click:
            pt_o = (float(click["x"])/scale, float(click["y"])/scale)
            if ss.tool == "CORA":  ss.cora  = pt_o
            if ss.tool == "HINGE": ss.hinge = pt_o
            ss.nonce += 1
            st.experimental_rerun()
    else:
        st.image(preview_img, width=cw)

# --- simulate movement if polygon is closed ---
if len(ss.poly) >= 3:
    center = ss.hinge or ss.cora
    if center:
        # build masks on ORIGINAL image
        m = polygon_mask(src_rgba.size, ss.poly)
        inv = ImageOps.invert(m)
        prox_img = Image.new("RGBA", src_rgba.size, (0,0,0,0)); prox_img.paste(src_rgba, (0,0), inv)
        dist_img = Image.new("RGBA", src_rgba.size, (0,0,0,0)); dist_img.paste(src_rgba, (0,0), m)

        moving = dist_img if ss.move_segment=="distal" else prox_img
        fixed  = prox_img if ss.move_segment=="distal" else dist_img

        moved = apply_affine(moving, ss.dx, ss.dy, ss.theta, center)
        out   = Image.alpha_composite(fixed, moved)

        # re-draw lines; each follows its own segment
        draw2 = ImageDraw.Draw(out)
        if len(ss.dist) == 2:
            pts = ss.dist if ss.move_segment=="proximal" else _transform_pts(ss.dist, ss.dx, ss.dy, ss.theta, center)
            draw2.line(pts, fill=(221,0,221,255), width=3)
        if len(ss.prox) == 2:
            pts = ss.prox if ss.move_segment=="distal" else _transform_pts(ss.prox, ss.dx, ss.dy, ss.theta, center)
            draw2.line(pts, fill=(66,133,244,255), width=3)

        # show simulated image resized to preview size
        st.image(out.resize((cw, ch), Image.NEAREST), caption="Simulation output", use_column_width=False)

# helper for line transform (original space)
def _transform_pts(points, dx, dy, angle_deg, center):
    ang = math.radians(angle_deg); c, s = math.cos(ang), math.sin(ang)
    cx, cy = center
    out = []
    for (x, y) in points:
        x0, y0 = x - cx, y - cy
        xr =  x0*c + y0*s + cx + dx
        yr = -x0*s + y0*c + cy + dy
        out.append((float(xr), float(yr)))
    return out

# -------- download parameters --------
params = {
    "polygon_points_original": ss.poly,
    "proximal_line_original": ss.prox,
    "distal_line_original": ss.dist,
    "cora_original": ss.cora,
    "hinge_original": ss.hinge,
    "move_segment": ss.move_segment,
    "dx": ss.dx, "dy": ss.dy, "rotate_deg": ss.theta,
}
st.download_button("Download parameters JSON",
                   data=io.BytesIO(str(params).encode("utf-8")).getvalue(),
                   file_name="osteotomy_params.json",
                   mime="application/json")
