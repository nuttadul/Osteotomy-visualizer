# app.py
import io, math, base64, json
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Osteotomy (Streamlit)", layout="wide")

# ---------------- helpers ----------------
def decode_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def polygon_mask(size, pts: List[Tuple[float, float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def centroid(pts):
    if len(pts) < 3:
        return None
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    a = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        cross = x[i] * y[j] - x[j] * y[i]
        a += cross
        cx += (x[i] + x[j]) * cross
        cy += (y[i] + y[j]) * cross
    a *= 0.5
    if abs(a) < 1e-9:
        return None
    cx /= (6 * a)
    cy /= (6 * a)
    return (cx, cy)

def apply_affine(img: Image.Image, dx, dy, rot_deg, center_xy):
    # Pillow rotates CCW in screen coords (y-down)
    rot = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def transform_points_screen(points, dx, dy, angle_deg, center):
    """Match Pillow rotation in screen coords (y-down)."""
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

def make_display_image(base_img: Image.Image, disp_w: int, state) -> tuple[Image.Image, float]:
    W, H = base_img.size
    scale = min(disp_w / float(W), 1.0)
    disp_h = int(round(H * scale))
    show = base_img.resize((int(round(W * scale)), disp_h), Image.NEAREST).copy()
    d = ImageDraw.Draw(show)

    # polygon (persisted)
    if state.poly:
        poly_disp = [(p[0] * scale, p[1] * scale) for p in state.poly]
        d.line(poly_disp, fill=(0, 255, 255, 255), width=2)
        if len(poly_disp) >= 3:
            d.line([*poly_disp, poly_disp[0]], fill=(0, 255, 255, 255), width=2)

    # persisted lines
    if len(state.prox) == 2:
        d.line(
            [
                (state.prox[0][0] * scale, state.prox[0][1] * scale),
                (state.prox[1][0] * scale, state.prox[1][1] * scale),
            ],
            fill=(66, 133, 244, 255),
            width=3,
        )
    if len(state.dist) == 2:
        d.line(
            [
                (state.dist[0][0] * scale, state.dist[0][1] * scale),
                (state.dist[1][0] * scale, state.dist[1][1] * scale),
            ],
            fill=(221, 0, 221, 255),
            width=3,
        )

    # centers
    if state.cora:
        x, y = state.cora
        x *= scale
        y *= scale
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 200, 0, 255), width=2)
    if state.hinge:
        x, y = state.hinge
        x *= scale
        y *= scale
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 165, 0, 255), width=3)
        d.line([(x - 12, y), (x + 12, y)], fill=(255, 165, 0, 255), width=1)
        d.line([(x, y - 12), (x, y + 12)], fill=(255, 165, 0, 255), width=1)

    return show, scale

# --- helpers to read canvas shapes back to original coords ---
def _parse_line(obj, scale):
    """
    Robustly recover absolute canvas coords for a Fabric 'line' object.
    Accounts for left/top anchor and object scaling.
    Returns original-image coords (dividing by scale).
    """
    ox = float(obj.get("left", 0.0))
    oy = float(obj.get("top", 0.0))
    sx = float(obj.get("scaleX", 1.0))
    sy = float(obj.get("scaleY", 1.0))
    x1 = ox + sx * float(obj.get("x1", 0.0))
    y1 = oy + sy * float(obj.get("y1", 0.0))
    x2 = ox + sx * float(obj.get("x2", 0.0))
    y2 = oy + sy * float(obj.get("y2", 0.0))
    return [(x1 / scale, y1 / scale), (x2 / scale, y2 / scale)]

def _parse_polygon(obj, scale):
    # try 'path' first (absolute path commands)
    if "path" in obj and isinstance(obj["path"], list):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M", "L"):
                pts.append((cmd[1] / scale, cmd[2] / scale))
        return pts
    # fallback to 'points' (+ left/top)
    if "points" in obj and isinstance(obj["points"], list):
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        sx = float(obj.get("scaleX", 1.0))
        sy = float(obj.get("scaleY", 1.0))
        pts = []
        for p in obj["points"]:
            px = left + sx * float(p.get("x", 0))
            py = top + sy * float(p.get("y", 0))
            pts.append((px / scale, py / scale))
        return pts
    return []

# ---------------- state ----------------
ss = st.session_state
defaults = dict(
    poly=[],
    cora=None,
    hinge=None,
    prox=[],
    dist=[],
    dispw=1100,
    dx=0,
    dy=0,
    theta=0,
    segment="distal",
    tool_prev=None,
    click_nonce=0,
    last_event=None,
)
for k, v in defaults.items():
    ss.setdefault(k, v)

st.sidebar.header("Upload image")
uploaded = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg", "tif", "tiff"])

tool = st.sidebar.radio(
    "Tool", ["Polygon", "CORA", "HINGE", "Prox line", "Dist line"], index=0
)

# reset click stream when tool switches (prevents ghost events)
if ss.tool_prev != tool:
    ss.click_nonce += 1
    ss.tool_prev = tool
    ss.last_event = None

ss.segment = st.sidebar.radio(
    "Move segment",
    ["distal", "proximal"],
    index=(0 if ss.segment == "distal" else 1),
    horizontal=True,
)
ss.dispw = st.sidebar.slider("Preview width", 600, 1800, ss.dispw, 50)
ss.dx = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)

c1, c2, c3, c4, c5 = st.sidebar.columns(5)
if c1.button("Undo"):
    if tool == "Polygon" and ss.poly:
        ss.poly.clear()  # polygon is multi-point; undo clears for simplicity
    elif tool == "Prox line" and ss.prox:
        ss.prox.clear()
    elif tool == "Dist line" and ss.dist:
        ss.dist.clear()
    elif tool == "CORA":
        ss.cora = None
    elif tool == "HINGE":
        ss.hinge = None
if c2.button("Reset poly"):
    ss.poly.clear()
if c3.button("Reset lines"):
    ss.prox.clear()
    ss.dist.clear()
if c4.button("Clear centers"):
    ss.cora = None
    ss.hinge = None
if c5.button("Reset move"):
    ss.dx = 0
    ss.dy = 0
    ss.theta = 0

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

img = decode_image(uploaded.getvalue())
st.markdown(
    "<style>.stImage img{cursor: crosshair !important;}</style>",
    unsafe_allow_html=True,
)

# ---------- base preview (used as background for tools) ----------
disp_img, scale = make_display_image(img, ss.dispw, ss)

def to_orig(pt_disp):
    return (float(pt_disp[0]) / scale, float(pt_disp[1]) / scale)

# ---------- CORA / HINGE click points ----------
if tool in ("CORA", "HINGE"):
    res = streamlit_image_coordinates(
        disp_img, width=disp_img.width, key=f"clicks-{ss.click_nonce}"
    )
    if res and "x" in res and "y" in res:
        event = (res["x"], res["y"], disp_img.width, disp_img.height)
        if event != ss.last_event:
            ss.last_event = event
            pt = to_orig((res["x"], res["y"]))
            if tool == "CORA":
                ss.cora = pt
            else:
                ss.hinge = pt

# ---------- LIVE tools (embed image via initial_drawing) ----------

def _png_data_url(pil_img: Image.Image) -> str:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    bio = BytesIO()
    pil_img.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")

def _fabric_image_scene(pil_img: Image.Image) -> dict:
    return {
        "version": "5.2.4",
        "objects": [
            {
                "type": "image",
                "left": 0,
                "top": 0,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "opacity": 1,
                "src": _png_data_url(pil_img),
                "selectable": False,
                "evented": False,
            }
        ],
    }

def _canvas_with_initial(bg_pil: Image.Image, **kwargs):
    W, H = bg_pil.size
    scene = json.dumps(_fabric_image_scene(bg_pil))

    def _call(**k):
        try:
            return st_canvas(display_ratio=1.0, **k)
        except TypeError:
            return st_canvas(**k)

    return _call(
        initial_drawing=scene,
        width=W,
        height=H,
        background_color=None,
        **kwargs,
    )

def _close_if_near_first(pts, scale, px_thresh=10):
    """If last vertex is within px_thresh (in display px) of first, snap to close."""
    if len(pts) < 3:
        return pts
    x0, y0 = pts[0]
    xn, yn = pts[-1]
    dx = (x0 - xn) * scale
    dy = (y0 - yn) * scale
    if (dx * dx + dy * dy) ** 0.5 <= px_thresh:
        pts[-1] = pts[0]
    return pts

def live_polygon(background_pil: Image.Image, existing_pts: list, key: str) -> list:
    try:
        result = _canvas_with_initial(
            background_pil,
            update_streamlit=True,
            drawing_mode="polygon",
            stroke_color="#00FFFF",
            stroke_width=2,
            fill_color="rgba(0,0,0,0)",
            key=key,
        )
    except Exception:
        result = _canvas_with_initial(
            background_pil,
            update_streamlit=True,
            drawing_mode="polyline",
            stroke_color="#00FFFF",
            stroke_width=2,
            fill_color="rgba(0,0,0,0)",
            key=key,
        )

    if result.json_data:
        objs = result.json_data.get("objects", [])
        for obj in reversed(objs):
            t = obj.get("type")
            if t == "polygon":
                pts = _parse_polygon(obj, scale)
                pts = _close_if_near_first(pts, scale)
                if len(pts) >= 3:
                    return pts
            if t in ("polyline", "path"):
                pts = _parse_polygon(obj, scale)
                pts = _close_if_near_first(pts, scale)
                if len(pts) >= 3:
                    return pts
    return existing_pts

def live_line(background_pil: Image.Image, current_line: list, color: str, key: str) -> list:
    result = _canvas_with_initial(
        background_pil,
        update_streamlit=True,
        drawing_mode="line",
        stroke_color=color,
        stroke_width=3,
        fill_color="rgba(0,0,0,0)",
        key=key,
    )
    if result.json_data:
        objs = result.json_data.get("objects", [])
        for obj in reversed(objs):
            if obj.get("type") == "line":
                return _parse_line(obj, scale)
    return current_line

# ---------- switch tools ----------
if tool == "Polygon":
    ss.poly = live_polygon(disp_img, ss.poly, key=f"canvas-poly-{ss.click_nonce}")

elif tool == "Prox line":
    ss.prox = live_line(disp_img, ss.prox, "#4285F4", key=f"canvas-prox-{ss.click_nonce}")

elif tool == "Dist line":
    ss.dist = live_line(disp_img, ss.dist, "#DD00DD", key=f"canvas-dist-{ss.click_nonce}")

# ---------- transform + display ----------
center = ss.hinge or ss.cora or centroid(ss.poly)
if len(ss.poly) >= 3 and center is not None:
    m = polygon_mask(img.size, ss.poly)
    inv = ImageOps.invert(m)
    prox_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    prox_img.paste(img, (0, 0), inv)
    dist_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dist_img.paste(img, (0, 0), m)

    moving = dist_img if ss.segment == "distal" else prox_img
    fixed = prox_img if ss.segment == "distal" else dist_img

    moved = apply_affine(moving, ss.dx, ss.dy, ss.theta, center)
    out = Image.alpha_composite(
        Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 0)), fixed), moved
    )

    # redraw lines; each follows its own segment
    draw2 = ImageDraw.Draw(out)
    if len(ss.dist) == 2:
        p = (
            transform_points_screen(ss.dist, ss.dx, ss.dy, ss.theta, center)
            if ss.segment == "distal"
            else ss.dist
        )
        draw2.line(p, fill=(221, 0, 221, 255), width=3)
    if len(ss.prox) == 2:
        p = (
            transform_points_screen(ss.prox, ss.dx, ss.dy, ss.theta, center)
            if ss.segment == "proximal"
            else ss.prox
        )
        draw2.line(p, fill=(66, 133, 244, 255), width=3)

    disp_out = out.resize((disp_img.width, disp_img.height), Image.NEAREST)
    st.image(disp_out, width=disp_img.width)

    # downloads
    params = dict(
        mode=ss.segment,
        dx=ss.dx,
        dy=ss.dy,
        rotate_deg=ss.theta,
        rotation_center=center,
        polygon_points=ss.poly,
        cora=ss.cora,
        hinge=ss.hinge,
        proximal_line=ss.prox,
        distal_line=ss.dist,
    )
    df = pd.DataFrame([params])
    st.download_button(
        "Download parameters CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="osteotomy_params.csv",
        mime="text/csv",
        key="csv",
    )
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button(
        "Download transformed image (PNG)",
        data=buf.getvalue(),
        file_name="osteotomy_transformed.png",
        mime="image/png",
        key="png",
    )
else:
    st.info(
        "Draw polygon (≥3) and set HINGE/CORA. Use Prox/Dist line tools to draw with a live preview (drag). "
        "Lines and polygon persist until you reset."
    )
