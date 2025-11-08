# app.py
import io, math
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# Streamlit setup
# =========================
st.set_page_config(page_title="Bone Ninja – one-pane osteotomy", layout="wide")


# =========================
# Utilities
# =========================
def safe_rerun():
    """Trigger a rerun and stop cleanly so we don't execute the rest of this cycle."""
    try:
        st.rerun()  # new API
    except Exception:
        st.experimental_rerun()  # legacy
    finally:
        st.stop()


def load_rgba(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img


def polygon_mask(size: Tuple[int, int], pts: List[Tuple[float, float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m


def centroid(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if len(pts) < 3:
        return None
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    a = 0.0; cx = 0.0; cy = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        cross = x[i] * y[j] - x[j] * y[i]
        a += cross
        cx += (x[i] + x[j]) * cross
        cy += (y[i] + y[j]) * cross
    a *= 0.5
    if abs(a) < 1e-9:
        return None
    cx /= (6 * a); cy /= (6 * a)
    return (cx, cy)


def apply_affine_fragment(img: Image.Image, dx, dy, rot_deg, center_xy):
    # Pillow rotates CCW with Y down; this matches “screen” coords
    rot = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out


def transform_points_y_down(points, dx, dy, angle_deg, center):
    """Rotate CCW about center in Y-down coords, then translate by (dx,dy)."""
    if not points:
        return []
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    cx, cy = center
    out = []
    for (x, y) in points:
        x0, y0 = x - cx, y - cy
        xr = x0 * c + y0 * s + cx + dx
        yr = -x0 * s + y0 * c + cy + dy
        out.append((float(xr), float(yr)))
    return out


def angle_deg(p0, p1):
    """Angle (deg) of vector p0->p1 in screen coords (x right, y down)."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return math.degrees(math.atan2(dy, dx))


def draw_label(d: ImageDraw.ImageDraw, text: str, x: float, y: float):
    d.rectangle([x - 4, y - 14, x + 150, y + 6], fill=(0, 0, 0, 160))
    d.text((x, y - 12), text, fill=(255, 255, 255, 230))


# =========================
# Session state
# =========================
ss = st.session_state
defaults = dict(
    disp_w=1100,
    # drawing state
    mode="Polygon",  # Polygon | Prox axis | Dist axis | Prox joint | Dist joint | HINGE | CORA
    snap_px=12,
    # stored (display coords)
    poly=[],
    poly_closed=False,
    prox_axis=[],
    dist_axis=[],
    prox_joint=[],
    dist_joint=[],
    hinge=None,
    cora=None,
    # pending (for 2-click lines)
    pending_first=None,  # (x,y) or None
    # move / simulate
    move_segment="distal",  # distal | proximal
    dx=0,
    dy=0,
    theta=0,
)
for k, v in defaults.items():
    ss.setdefault(k, v)


# =========================
# Sidebar
# =========================
st.sidebar.header("Upload X-ray")
up = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg", "tif", "tiff"])

ss.mode = st.sidebar.radio(
    "Tool",
    ["Polygon", "Prox axis", "Dist axis", "Prox joint", "Dist joint", "HINGE", "CORA"],
    index=["Polygon", "Prox axis", "Dist axis", "Prox joint", "Dist joint", "HINGE", "CORA"].index(ss.mode),
)

ss.move_segment = st.sidebar.radio("Move which segment?", ["distal", "proximal"], index=(0 if ss.move_segment == "distal" else 1))

ss.disp_w = st.sidebar.slider("Display width", 600, 1800, ss.disp_w, 50)
ss.snap_px = st.sidebar.slider("Polygon snap (px)", 4, 30, int(ss.snap_px), 1)

st.sidebar.markdown("---")
col_reset1, col_reset2, col_reset3, col_reset4 = st.sidebar.columns(4)
if col_reset1.button("Reset poly"):
    ss.poly.clear(); ss.poly_closed = False; ss.pending_first = None
if col_reset2.button("Reset lines"):
    ss.prox_axis.clear(); ss.dist_axis.clear(); ss.prox_joint.clear(); ss.dist_joint.clear(); ss.pending_first = None
if col_reset3.button("Clear points"):
    ss.hinge = None; ss.cora = None
if col_reset4.button("Clear all"):
    for k in ("poly", "prox_axis", "dist_axis", "prox_joint", "dist_joint"):
        ss[k] = []
    ss.poly_closed = False; ss.hinge = None; ss.cora = None; ss.pending_first = None

st.sidebar.markdown("---")
ss.dx = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)


# =========================
# Load + scale image
# =========================
if not up:
    st.info("Upload an X-ray to begin.")
    st.stop()

img_rgba = load_rgba(up.getvalue())
W, H = img_rgba.size
scale = min(ss.disp_w / float(W), 1.0)
disp_h = int(round(H * scale))
disp_img = img_rgba.resize((int(round(W * scale)), disp_h), Image.NEAREST)


# =========================
# Click handling (single pane)
# =========================
# We always show the composed preview, then capture a click on it.
def capture_click_on(image: Image.Image, key: str = "click-main"):
    return streamlit_image_coordinates(image, width=image.width, key=key)


# =========================
# Compose image (simulate first, then draw overlay)
# =========================
poly_disp = ss.poly[:]  # already stored in display coordinates

# Build the osteotomy preview bitmap (without lines yet)
composite = disp_img.copy()

center = None
if ss.poly_closed and len(poly_disp) >= 3:
    mask = polygon_mask(disp_img.size, poly_disp)
    inv = ImageOps.invert(mask)

    prox = Image.new("RGBA", disp_img.size, (0, 0, 0, 0)); prox.paste(disp_img, (0, 0), inv)
    dist = Image.new("RGBA", disp_img.size, (0, 0, 0, 0)); dist.paste(disp_img, (0, 0), mask)

    moving = dist if ss.move_segment == "distal" else prox
    fixed = prox if ss.move_segment == "distal" else dist

    center = ss.hinge or centroid(poly_disp) or (disp_img.width / 2.0, disp_img.height / 2.0)

    moved = apply_affine_fragment(moving, ss.dx, ss.dy, ss.theta, center)

    composite = Image.new("RGBA", disp_img.size, (0, 0, 0, 0))
    composite.alpha_composite(fixed)
    composite.alpha_composite(moved)

# Prepare **transformed copies** of lines (never mutate session state)
prox_axis_draw = ss.prox_axis[:]
dist_axis_draw = ss.dist_axis[:]
prox_joint_draw = ss.prox_joint[:]
dist_joint_draw = ss.dist_joint[:]

if ss.poly_closed and len(poly_disp) >= 3:
    c = center or (disp_img.width / 2.0, disp_img.height / 2.0)
    if ss.move_segment == "distal":
        if len(dist_axis_draw) == 2:
            dist_axis_draw = transform_points_y_down(dist_axis_draw, ss.dx, ss.dy, ss.theta, c)
        if len(dist_joint_draw) == 2:
            dist_joint_draw = transform_points_y_down(dist_joint_draw, ss.dx, ss.dy, ss.theta, c)
    else:
        if len(prox_axis_draw) == 2:
            prox_axis_draw = transform_points_y_down(prox_axis_draw, ss.dx, ss.dy, ss.theta, c)
        if len(prox_joint_draw) == 2:
            prox_joint_draw = transform_points_y_down(prox_joint_draw, ss.dx, ss.dy, ss.theta, c)


# Draw overlay (polygon, nodes, axes, joint lines, points, angle labels)
overlay = composite.copy()
d = ImageDraw.Draw(overlay, "RGBA")

# polygon with live nodes
if poly_disp:
    if len(poly_disp) >= 2:
        d.line(poly_disp, fill=(0, 255, 255, 255), width=2)
    if ss.poly_closed and len(poly_disp) >= 3:
        d.line([poly_disp[-1], poly_disp[0]], fill=(0, 255, 255, 255), width=2)
    for p in poly_disp:
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(0, 255, 255, 200))

def _draw_line(line, color):
    if len(line) == 2:
        d.line(line, fill=color, width=3)
        for p in line:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=color)

_draw_line(prox_axis_draw, (66, 133, 244, 255))
_draw_line(dist_axis_draw, (221, 0, 221, 255))
_draw_line(prox_joint_draw, (255, 215, 0, 220))
_draw_line(dist_joint_draw, (255, 215, 0, 220))

# points
if ss.cora:
    x, y = ss.cora
    d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 200, 0, 255), width=2)
if ss.hinge:
    x, y = ss.hinge
    d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 165, 0, 255), width=3)
    d.line([(x - 12, y), (x + 12, y)], fill=(255, 165, 0, 255), width=1)
    d.line([(x, y - 12), (x, y + 12)], fill=(255, 165, 0, 255), width=1)

# angle labels
yl = 8
for L in (prox_joint_draw, dist_joint_draw, prox_axis_draw, dist_axis_draw):
    if len(L) == 2:
        a = angle_deg(L[0], L[1])
        draw_label(d, f"angle {a:.1f}°", 8, yl)
        yl += 18


# =========================
# Show + capture click
# =========================
st.image(overlay, width=overlay.width)  # one pane only
click = capture_click_on(overlay)       # returns {'x':..., 'y':...} or None


# =========================
# Update state from click (IMMEDIATE reaction)
# =========================
def add_point(line_attr: str, pt: Tuple[float, float]):
    """2-click lines: store first in ss.pending_first, second closes and writes to attr."""
    if ss.pending_first is None:
        ss.pending_first = pt
    else:
        p0 = ss.pending_first
        setattr(ss, line_attr, [p0, pt])
        ss.pending_first = None
    safe_rerun()


if click and "x" in click and "y" in click:
    pt = (float(click["x"]), float(click["y"]))

    if ss.mode == "Polygon":
        # snap to first?
        if ss.poly and ( (pt[0]-ss.poly[0][0])**2 + (pt[1]-ss.poly[0][1])**2 )**0.5 <= ss.snap_px and len(ss.poly) >= 2:
            ss.poly.append(ss.poly[0])  # close visually
            ss.poly_closed = True
        else:
            ss.poly.append(pt)
        safe_rerun()

    elif ss.mode == "Prox axis":
        add_point("prox_axis", pt)

    elif ss.mode == "Dist axis":
        add_point("dist_axis", pt)

    elif ss.mode == "Prox joint":
        add_point("prox_joint", pt)

    elif ss.mode == "Dist joint":
        add_point("dist_joint", pt)

    elif ss.mode == "HINGE":
        ss.hinge = pt; safe_rerun()

    elif ss.mode == "CORA":
        ss.cora = pt; safe_rerun()


# =========================
# Small footer help
# =========================
with st.expander("Help / What to click"):
    st.markdown(
        """
- **Polygon**: click to drop vertices. Click near the **first** point to snap and close the cut.  
- **Prox/Dist axis / Prox/Dist joint**: two clicks (start → end). A node appears each click.  
- **HINGE / CORA**: click once to set.  
- Use **ΔX/ΔY/Rotate** to move the selected segment (**distal** or **proximal**).  
- Lines rotate **with** their segment; internal data aren’t mutated, only the display copies are transformed.
        """
    )
