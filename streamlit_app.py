# app.py
# -------------------------------
# Osteotomy visualizer (click-based, no canvas)
# - Upload an X-ray
# - Draw joints, axes (with optional angle assist + ghost line)
# - Draw polygon and set hinge
# - Simulate distal/proximal movement around hinge
# -------------------------------

import io, math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(
    page_title="Osteotomy Visualizer (click-based)",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* nicer radio spacing */
    div[role="radiogroup"] > label { margin-right: .75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# Helpers
# -------------------------------
def load_rgba(file_bytes: bytes) -> Image.Image:
    """Load image, honor EXIF orientation, RGBA mode."""
    img = Image.open(io.BytesIO(file_bytes))
    return ImageOps.exif_transpose(img).convert("RGBA")


def polygon_mask(size: Tuple[int, int], pts: List[Tuple[float, float]]) -> Image.Image:
    """Create 8-bit mask (L) with a filled polygon."""
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m


def rotate_about(pt: Tuple[float, float], center: Tuple[float, float], deg: float) -> Tuple[float, float]:
    """Rotate a point about center in screen coords (y-down)."""
    x, y = pt
    cx, cy = center
    ang = math.radians(deg)
    c, s = math.cos(ang), math.sin(ang)
    # Screen coords: positive rotation is CCW visually
    x0, y0 = x - cx, y - cy
    xr = x0 * c + y0 * s
    yr = -x0 * s + y0 * c
    return (xr + cx, yr + cy)


def apply_osteotomy(
    src_rgba: Image.Image,
    poly_pts: List[Tuple[float, float]],
    hinge: Tuple[float, float],
    dx: float,
    dy: float,
    rot_deg: float,
    segment: str = "distal",
) -> Image.Image:
    """
    Split image by polygon; transform chosen segment (distal=inside polygon) by dx,dy,rot about hinge.
    Return composited RGBA.
    """
    W, H = src_rgba.size
    mask = polygon_mask((W, H), poly_pts)
    inv_mask = ImageOps.invert(mask)

    # inside = distal; outside = proximal
    inside = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    outside = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    inside.paste(src_rgba, (0, 0), mask)
    outside.paste(src_rgba, (0, 0), inv_mask)

    moving = inside if segment == "distal" else outside
    fixed  = outside if segment == "distal" else inside

    # rotate around hinge, then translate
    rot = moving.rotate(rot_deg, resample=Image.BICUBIC, center=hinge, expand=False)

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(fixed, (0, 0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out


def dist2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def line_angle_deg(p0, p1) -> float:
    """Return 0..180 line angle (screen coords)."""
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180
    return ang


def snap_endpoint_to_angle(p0, p1, target_deg, tol_deg) -> Tuple[float, float]:
    """
    If p1 is within 'tol_deg' of target_deg (or target+180), snap to that angle keeping the same length.
    Otherwise return original p1.
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx == 0 and dy == 0:
        return p1

    cur = line_angle_deg(p0, p1)
    # pick nearest (target or target+180)
    choices = [target_deg % 180, (target_deg + 180) % 180]
    best = min(choices, key=lambda a: min(abs(a - cur), 180 - abs(a - cur)))
    delta = min(abs(best - cur), 180 - abs(best - cur))
    if delta > tol_deg:
        return p1

    L = math.hypot(dx, dy)
    ang = math.radians(best)
    return (p0[0] + L * math.cos(ang), p0[1] + L * math.sin(ang))


# -------------------------------
# Session state model
# -------------------------------
ss = st.session_state
defaults = dict(
    tool="Polygon",               # "Joints", "Axes", "Polygon", "Hinge", "Simulate"
    subtool="Add",                # for joints/axes
    dispw=1100,                   # display width
    joints=[],                    # [(x,y)]
    axes=[],                      # [{"joint": int|None, "p0":(x,y), "p1":(x,y), "label":str}]
    placing_axis=None,            # {"joint": idx|None, "p0":(x,y)}
    poly=[],                      # polygon points (original coords)
    hinge=None,                   # (x,y)
    snap_on=False,
    snap_deg=81.0,
    snap_tol=3.0,
    hud_text="",                  # live readout
    click_nonce=0,                # to flush stale clicks when switching tools
)
for k, v in defaults.items():
    ss.setdefault(k, v)


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Upload image")
uploaded = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg", "tif", "tiff"])

st.sidebar.markdown("### Display")
ss.dispw = st.sidebar.slider("Preview width", 600, 1800, int(ss.dispw), 50)

st.sidebar.markdown("### Angle assistance (for axes)")
ss.snap_on = st.sidebar.checkbox("Constrain axis to angle", value=bool(ss.snap_on))
ss.snap_deg = float(st.sidebar.number_input("Target angle (deg)", value=float(ss.snap_deg), step=0.5, format="%.1f"))
ss.snap_tol = float(st.sidebar.slider("Tolerance (deg)", 0.0, 15.0, float(ss.snap_tol), 0.5))

hud = st.sidebar.empty()
if ss.hud_text:
    hud.write(ss.hud_text)

cols_btn = st.sidebar.columns(3)
if cols_btn[0].button("Reset poly"):
    ss.poly.clear()
if cols_btn[1].button("Reset axes"):
    ss.axes.clear(); ss.placing_axis = None
if cols_btn[2].button("Reset joints/hinge"):
    ss.joints.clear(); ss.hinge = None


# -------------------------------
# Guard: need an image
# -------------------------------
if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

img = load_rgba(uploaded.getvalue())
origW, origH = img.size
scale = min(ss.dispw / float(origW), 1.0)
dispH = int(round(origH * scale))
dispW = int(round(origW * scale))


def o2c(p):  # original -> canvas
    return (p[0] * scale, p[1] * scale)


def c2o(p):  # canvas -> original
    return (p[0] / scale, p[1] / scale)


# -------------------------------
# Tool ribbon (always visible)
# -------------------------------
left, right = st.columns([1.05, 1])

r0 = left.columns([1.2, 1, 1, 1, 1])
left.subheader("Live Drawing")

ss.tool = r0[0].radio(
    "Tool",
    ["Joints", "Axes", "Polygon", "Hinge", "Simulate"],
    index=["Joints", "Axes", "Polygon", "Hinge", "Simulate"].index(ss.tool),
    horizontal=True,
    label_visibility="collapsed",
)
if ss.tool in ("Joints", "Axes"):
    ss.subtool = r0[1].radio(
        "Action",
        ["Add", "Delete"],
        index=["Add", "Delete"].index(ss.subtool),
        horizontal=True,
        label_visibility="collapsed",
    )
else:
    r0[1].markdown("&nbsp;")

right.subheader("Preview / Simulation")


# -------------------------------
# Build overlay (left panel)
# -------------------------------
overlay = img.resize((dispW, dispH), Image.NEAREST).copy()
draw = ImageDraw.Draw(overlay, "RGBA")

# Draw persisted polygon
if len(ss.poly) >= 2:
    pts_c = [o2c(p) for p in ss.poly]
    draw.line(pts_c, fill=(0, 255, 255, 255), width=2)
    if len(ss.poly) >= 3 and ss.poly[0] == ss.poly[-1]:
        draw.polygon(pts_c, outline=(0, 255, 255, 255), fill=(0, 255, 255, 35))

# Draw joints
for (jx, jy) in ss.joints:
    x, y = o2c((jx, jy))
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 215, 0, 255), width=2)

# Draw axes
for ax in ss.axes:
    draw.line([o2c(ax["p0"]), o2c(ax["p1"])], fill=(66, 133, 244, 255), width=3)

# Draw hinge
if ss.hinge:
    x, y = o2c(ss.hinge)
    draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 165, 0, 255), width=3)
    draw.line([(x - 12, y), (x + 12, y)], fill=(255, 165, 0, 255), width=1)
    draw.line([(x, y - 12), (x, y + 12)], fill=(255, 165, 0, 255), width=1)

# Ghost (axis placement)
if ss.tool == "Axes" and ss.placing_axis is not None:
    p0 = ss.placing_axis["p0"]
    # draw a faint crosshair on the origin
    cx, cy = o2c(p0)
    draw.line([(cx - 10, cy), (cx + 10, cy)], fill=(180, 180, 255, 200), width=1)
    draw.line([(cx, cy - 10), (cx, cy + 10)], fill=(180, 180, 255, 200), width=1)

# Display + capture click (always RGB here to avoid JPEG plugin errors)
click = streamlit_image_coordinates(overlay.convert("RGB"), width=dispW, key=f"click-{ss.click_nonce}")

xo = yo = None
if click and "x" in click and "y" in click:
    xo, yo = c2o((float(click["x"]), float(click["y"])))


# -------------------------------
# Handle clicks for each tool
# -------------------------------
# Note: we never draw any blocking tooltip on the image;
# live data goes to the sidebar HUD (hud.write).

if ss.tool == "Joints":
    if ss.subtool == "Add" and xo is not None:
        ss.joints.append((xo, yo))
    elif ss.subtool == "Delete" and xo is not None and ss.joints:
        j = int(np.argmin([dist2((xo, yo), p) for p in ss.joints]))
        if math.hypot(xo - ss.joints[j][0], yo - ss.joints[j][1]) < 25 / scale:
            ss.joints.pop(j)

elif ss.tool == "Axes":
    if ss.subtool == "Delete" and xo is not None and ss.axes:
        # delete nearest axis by distance to its midpoint
        mids = [((ax["p0"][0] + ax["p1"][0]) / 2, (ax["p0"][1] + ax["p1"][1]) / 2) for ax in ss.axes]
        k = int(np.argmin([dist2((xo, yo), m) for m in mids]))
        if math.hypot(xo - mids[k][0], yo - mids[k][1]) < 30 / scale:
            ss.axes.pop(k)
    else:
        # Add: two clicks, with ghost and angle assist
        if ss.placing_axis is None and xo is not None:
            # bind to nearest joint if inside 30 px
            bind_idx = None
            if ss.joints:
                j = int(np.argmin([dist2((xo, yo), p) for p in ss.joints]))
                if math.hypot(xo - ss.joints[j][0], yo - ss.joints[j][1]) < 30 / scale:
                    bind_idx = j
                    p0 = ss.joints[j]
                else:
                    p0 = (xo, yo)
            else:
                p0 = (xo, yo)
            ss.placing_axis = {"joint": bind_idx, "p0": p0}

        elif ss.placing_axis is not None:
            p0 = ss.placing_axis["p0"]
            # ghost endpoint follows cursor; snap if enabled
            # We use current mouse position as ghost even if click is None
            # so compute mouse position from last image-coordinates payload
            if click and "x" in click and "y" in click:
                ghost = (xo, yo)
                if ss.snap_on:
                    ghost = snap_endpoint_to_angle(p0, ghost, ss.snap_deg, ss.snap_tol)
                # draw ghost on top
                g0, g1 = o2c(p0), o2c(ghost)
                g_overlay = img.resize((dispW, dispH), Image.NEAREST).copy()
                g_draw = ImageDraw.Draw(g_overlay, "RGBA")
                g_draw.line([g0, g1], fill=(66, 133, 244, 200), width=3)
                # show angle readout
                hud.write(
                    f"axis angle **{line_angle_deg(p0, ghost):.1f}°**, length **{math.hypot(ghost[0]-p0[0], ghost[1]-p0[1]):.0f}px**"
                )
                # replace the left image temporarily (no extra click needed)
                left.image(g_overlay, use_column_width=False, width=dispW)
            # confirm on actual click event
            if xo is not None:
                end = (xo, yo)
                if ss.snap_on:
                    end = snap_endpoint_to_angle(p0, end, ss.snap_deg, ss.snap_tol)
                ss.axes.append({"joint": ss.placing_axis["joint"], "p0": p0, "p1": end, "label": f"AX{len(ss.axes)+1}"})
                ss.placing_axis = None

elif ss.tool == "Polygon":
    # click to add; snap-close if near first point
    if xo is not None:
        pts = ss.poly + [(xo, yo)]
        if len(pts) >= 3 and math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < 12 / scale:
            pts[-1] = pts[0]  # close loop
        ss.poly = pts

elif ss.tool == "Hinge":
    if xo is not None:
        ss.hinge = (xo, yo)

# If a tool change happened earlier, this prevents reusing a stale click
# (not strictly necessary here but kept for safety)
if ss.tool in ("Joints", "Axes", "Polygon", "Hinge"):
    ss.click_nonce += 0  # no-op to make code explicit


# -------------------------------
# Right panel: Preview / Simulation
# -------------------------------
if ss.tool == "Simulate" and len(ss.poly) >= 3 and ss.poly[0] == ss.poly[-1] and ss.hinge:
    segment = right.radio("Move segment", ["distal", "proximal"], horizontal=True)
    dx = right.slider("ΔX (px)", -1000, 1000, 0, 1)
    dy = right.slider("ΔY (px)", -1000, 1000, 0, 1)
    theta = right.slider("Rotate (deg)", -180, 180, 0, 1)
    out = apply_osteotomy(img, ss.poly, ss.hinge, dx, dy, theta, segment)
    right.image(out.resize((dispW, dispH), Image.NEAREST), use_column_width=False)
else:
    # show same overlay (no blocking labels)
    right.image(overlay, use_column_width=False, width=dispW)
