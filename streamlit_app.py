# app.py  — Bone-ninja style click drawing (single-image), no canvas component.
# Patched: always pass RGB (not RGBA) into streamlit_image_coordinates to avoid JPEG OSError.

import io, math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


# -------------------------- small helpers --------------------------

def load_rgba(file_bytes: bytes) -> Image.Image:
    """Load an uploaded image as RGBA, respecting EXIF orientation."""
    im = Image.open(io.BytesIO(file_bytes))
    return ImageOps.exif_transpose(im).convert("RGBA")

def draw_nodes(draw: ImageDraw.ImageDraw, pts: List[Tuple[float,float]], r=4, color=(0,255,255,255)):
    for x,y in pts:
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)

def line_angle_deg(p0, p1) -> float:
    """Angle (deg) of the line p0->p1 in math coords (x right, y down), returned in [0,180)."""
    dx, dy = (p1[0]-p0[0], p1[1]-p0[1])
    if dx == 0 and dy == 0:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx))  # y-down
    if ang < 0: ang += 180.0
    return ang

def polygon_mask(size, pts: List[Tuple[float,float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255)
    return m

def apply_affine_rgba(img: Image.Image, dx: float, dy: float, rot_deg: float, center_xy: Tuple[float,float]) -> Image.Image:
    """
    Rotate around center (screen coords: y down), then translate by (dx,dy).
    PIL rotates counterclockwise for y-up; for y-down, simply use +rot_deg (empirically matches overlay).
    """
    rotated = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", img.size, (0,0,0,0))
    out.alpha_composite(rotated, (int(round(dx)), int(round(dy))))
    return out

def transform_points_screen(points, dx, dy, angle_deg, center):
    """Rotate + translate a polyline in y-down screen coords so it follows the same transform as the image."""
    if not points: return []
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    cx, cy = center
    out = []
    for (x, y) in points:
        x0, y0 = x - cx, y - cy
        xr = x0*c - y0*s + cx + dx
        yr = x0*s + y0*c + cy + dy
        out.append((xr, yr))
    return out

def text_bg(draw, xy, text, fg=(0,0,0), bg=(255,255,255,200), pad=3):
    w, h = draw.textsize(text)
    x, y = xy
    draw.rectangle((x-pad, y-pad, x+w+pad, y+h+pad), fill=bg)
    draw.text((x, y), text, fill=fg)


# -------------------------- Streamlit UI --------------------------

st.set_page_config(page_title="Osteotomy (click tools, no-canvas, RGB-patched)", layout="wide")

ss = st.session_state
defaults = dict(
    tool="Polygon",                 # current tool
    side="distal",                  # which segment moves
    dispw=1100,                     # display width
    dx=0, dy=0, theta=0,            # transform
    poly=[],                        # polygon vertices (orig px)
    hinge=None,                     # hinge point (orig px)
    cora=None,                      # CORA point (optional)
    prox=[], dist=[],               # axes (2 points each) in orig px
    joint=[],                       # joint tangent 2-pt line (orig px)
    first_click=None,               # pending first point for 2-pt tools
    snap_px=10,                     # close polygon within this display px
    _click_nonce=0,                 # bump to force component remount if needed
)
for k, v in defaults.items():
    ss.setdefault(k, v)

st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

ss.tool = st.sidebar.radio(
    "Tool",
    ["Polygon", "Hinge", "CORA", "Joint line", "Prox axis", "Dist axis", "Measure"],
    index=["Polygon","Hinge","CORA","Joint line","Prox axis","Dist axis","Measure"].index(ss.tool) if ss.tool in ["Polygon","Hinge","CORA","Joint line","Prox axis","Dist axis","Measure"] else 0
)

ss.side = st.sidebar.radio("Move segment", ["distal","proximal"], index=(0 if ss.side=="distal" else 1), horizontal=True)

ss.dispw = st.sidebar.slider("Display width", 600, 2000, ss.dispw, 50)
col1, col2, col3 = st.sidebar.columns(3)
with col1: reset_poly = st.button("Reset poly")
with col2: reset_axes = st.button("Reset axes")
with col3: reset_pts  = st.button("Reset hinge/CORA")

if reset_poly:
    ss.poly = []
if reset_axes:
    ss.prox, ss.dist, ss.joint = [], [], []
if reset_pts:
    ss.hinge, ss.cora = None, None

st.sidebar.divider()
st.sidebar.caption("Transform (applies after polygon + hinge):")
ss.dx    = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy    = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# -------------------------- Image + Scale --------------------------

orig = load_rgba(up.getvalue())        # RGBA
W, H = orig.size
scale = min(ss.dispw / float(W), 1.0)
disp_size = (int(round(W*scale)), int(round(H*scale)))

def o2c(pt): return (pt[0]*scale, pt[1]*scale)           # original->canvas/display
def c2o(pt): return (pt[0]/scale, pt[1]/scale)           # display->original

# -------------------------- Capture click (RGB PATCH) --------------------------

# Build the image we’ll show and click on (with current overlay drawn first).
base = orig.copy()

# If we have a valid osteotomy + hinge, transform the selected segment first so the user
# sees what will be exported.
center_for_rot = ss.hinge if ss.hinge else (W/2, H/2)
out = base.copy()
if len(ss.poly) >= 3 and ss.hinge:
    m = polygon_mask((W,H), ss.poly)
    inv = ImageOps.invert(m)
    prox_img = Image.new("RGBA", (W,H), (0,0,0,0)); prox_img.paste(base, (0,0), inv)
    dist_img = Image.new("RGBA", (W,H), (0,0,0,0)); dist_img.paste(base, (0,0), m)

    moving = dist_img if ss.side=="distal" else prox_img
    fixed  = prox_img if ss.side=="distal" else dist_img

    moved = apply_affine_rgba(moving, ss.dx, ss.dy, ss.theta, center_for_rot)
    out = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", (W,H), (0,0,0,0)), fixed), moved)

# Now draw overlay (polygon, axes, hinge/CORA, joint line) onto `out` so user clicks on actual view.
disp = out.copy()
d = ImageDraw.Draw(disp, "RGBA")

# polygon
if len(ss.poly) >= 2:
    # edge + nodes (cyan)
    d.line(ss.poly, fill=(0,255,255,180), width=2)
    if len(ss.poly) >= 3:
        d.line([*ss.poly, ss.poly[0]], fill=(0,255,255,180), width=2)
    draw_nodes(d, ss.poly, r=4, color=(0,255,255,255))

# axes
if len(ss.prox) == 2: d.line(ss.prox, fill=(66,133,244,255), width=3)    # blue
if len(ss.dist) == 2: d.line(ss.dist, fill=(221,0,221,255), width=3)     # magenta
if len(ss.joint) == 2: d.line(ss.joint, fill=(255,255,0,220), width=3)   # yellow

# hinge/cora
if ss.hinge:
    x,y = ss.hinge
    d.ellipse((x-6,y-6,x+6,y+6), outline=(255,165,0,255), width=3)  # orange
if ss.cora:
    x,y = ss.cora
    d.ellipse((x-5,y-5,x+5,y+5), outline=(0,200,0,255), width=2)    # green

# live angle read-outs if joint + axis exist
if len(ss.joint) == 2:
    j_ang = line_angle_deg(*ss.joint)
    text_bg(d, (10,10), f"Joint angle: {j_ang:.1f}°")
    if len(ss.prox) == 2:
        a = abs(line_angle_deg(*ss.prox) - j_ang); a = min(a, 180-a)
        text_bg(d, (10, 30), f"Prox vs Joint: {a:.1f}°")
    if len(ss.dist) == 2:
        a = abs(line_angle_deg(*ss.dist) - j_ang); a = min(a, 180-a)
        text_bg(d, (10, 50), f"Dist vs Joint: {a:.1f}°")

# Convert to display size for fast drawing and click coordinate capture
disp = disp.resize(disp_size, Image.NEAREST)

# IMPORTANT PATCH: pass RGB, not RGBA, to streamlit_image_coordinates (it saves JPEG internally)
disp_rgb = disp.convert("RGB")

# Show the one working image
st.image(disp_rgb, use_column_width=False)

# capture one click (returns dict or None)
click = streamlit_image_coordinates(disp_rgb, width=disp_size[0], key=f"click-{ss._click_nonce}")

# -------------------------- Handle click by active tool --------------------------

def commit_rerun():
    # Try Streamlit's modern rerun (safe if available), but keep UI stable if not.
    try:
        st.rerun()
    except Exception:
        pass

if click and "x" in click and "y" in click:
    pt_o = c2o((float(click["x"]), float(click["y"])))  # convert to original px

    if ss.tool == "Polygon":
        # close if near first; otherwise append
        if len(ss.poly) >= 2:
            first = ss.poly[0]
            dist2 = (pt_o[0]-first[0])**2 + (pt_o[1]-first[1])**2
            if dist2 <= (ss.snap_px/scale)**2 and len(ss.poly) >= 3:
                ss.poly.append(first)  # close exactly
            else:
                ss.poly.append(pt_o)
        else:
            ss.poly.append(pt_o)

    elif ss.tool in ("Prox axis","Dist axis","Joint line"):
        target = "prox" if ss.tool=="Prox axis" else ("dist" if ss.tool=="Dist axis" else "joint")
        # 2-click line
        if ss.first_click is None:
            ss.first_click = pt_o
        else:
            seg = [ss.first_click, pt_o]
            setattr(ss, target, seg)
            ss.first_click = None

    elif ss.tool == "Hinge":
        ss.hinge = pt_o

    elif ss.tool == "CORA":
        ss.cora = pt_o

    elif ss.tool == "Measure":
        # A simple 2-click temp measure line parked in prox slot
        if ss.first_click is None:
            ss.first_click = pt_o
        else:
            ss.prox = [ss.first_click, pt_o]
            ss.first_click = None

    # immediate refresh so the new node/line appears without a second click
    commit_rerun()


# -------------------------- Make the transformed result + downloads --------------------------

st.divider()
st.caption("Export (applies to current transform & overlay)")

if len(ss.poly) >= 3 and ss.hinge:
    # Build transformed composite again (same as preview)
    base = orig.copy()
    m = polygon_mask((W,H), ss.poly)
    inv = ImageOps.invert(m)
    prox_img = Image.new("RGBA", (W,H), (0,0,0,0)); prox_img.paste(base, (0,0), inv)
    dist_img = Image.new("RGBA", (W,H), (0,0,0,0)); dist_img.paste(base, (0,0), m)

    moving = dist_img if ss.side=="distal" else prox_img
    fixed  = prox_img if ss.side=="distal" else dist_img

    moved = apply_affine_rgba(moving, ss.dx, ss.dy, ss.theta, ss.hinge)
    export_rgba = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", (W,H), (0,0,0,0)), fixed), moved)

    # redraw axes to follow their own segment
    draw2 = ImageDraw.Draw(export_rgba)
    if len(ss.dist) == 2:
        p = transform_points_screen(ss.dist, ss.dx, ss.dy, ss.theta, ss.hinge) if ss.side=="distal" else ss.dist
        draw2.line(p, fill=(221,0,221,255), width=3)
    if len(ss.prox) == 2:
        p = transform_points_screen(ss.prox, ss.dx, ss.dy, ss.theta, ss.hinge) if ss.side=="proximal" else ss.prox
        draw2.line(p, fill=(66,133,244,255), width=3)

    # polygon outline
    if len(ss.poly) >= 3:
        draw2.line([*ss.poly, ss.poly[0]], fill=(0,255,255,180), width=2)

    # show a scaled export preview below (optional)
    st.image(export_rgba.resize(disp_size, Image.NEAREST).convert("RGB"), caption="Transformed result", use_column_width=False)

    # download
    buf = io.BytesIO()
    export_rgba.save(buf, format="PNG")
    st.download_button(
        "Download transformed image (PNG)",
        data=buf.getvalue(),
        file_name="osteotomy_transformed.png",
        mime="image/png"
    )
else:
    st.info("To transform: draw a polygon (≥3 points) and set a hinge. Then use ΔX/ΔY/Rotate. Axes follow their segment automatically.")
