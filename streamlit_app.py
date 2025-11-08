import io, math, json
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

# Canvas component
from streamlit_drawable_canvas import st_canvas  # live, interactive

st.set_page_config(page_title="Osteotomy (Canvas)", layout="wide")

# ============================ helpers ============================

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

def apply_affine(img: Image.Image, dx, dy, rot_deg, center_xy):
    rot = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(rot, (int(round(dx)), int(round(dy))))
    return out

def transform_points_screen(points, dx, dy, angle_deg, center):
    if not points: return []
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

def pil_to_np_rgb(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)

def draw_persisted_overlay(img: Image.Image, scale: float, ss) -> Image.Image:
    """Return a copy of img with persisted poly/lines/points drawn (scaled)."""
    show = img.copy()
    d = ImageDraw.Draw(show)

    # polygon (persist)
    if ss.poly:
        poly_disp = [(p[0] * scale, p[1] * scale) for p in ss.poly]
        if len(poly_disp) >= 2:
            d.line(poly_disp, fill=(0, 255, 255, 255), width=2)
        if len(poly_disp) >= 3:
            d.line([*poly_disp, poly_disp[0]], fill=(0, 255, 255, 255), width=2)

    # persisted lines
    if len(ss.prox) == 2:
        d.line([(ss.prox[0][0] * scale, ss.prox[0][1] * scale),
                (ss.prox[1][0] * scale, ss.prox[1][1] * scale)],
               fill=(66, 133, 244, 255), width=3)
    if len(ss.dist) == 2:
        d.line([(ss.dist[0][0] * scale, ss.dist[0][1] * scale),
                (ss.dist[1][0] * scale, ss.dist[1][1] * scale)],
               fill=(221, 0, 221, 255), width=3)

    # centers
    if ss.cora:
        x, y = ss.cora; x *= scale; y *= scale
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 200, 0, 255), width=2)
    if ss.hinge:
        x, y = ss.hinge; x *= scale; y *= scale
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 165, 0, 255), width=3)
        d.line([(x - 12, y), (x + 12, y)], fill=(255, 165, 0, 255), width=1)
        d.line([(x, y - 12), (x, y + 12)], fill=(255, 165, 0, 255), width=1)

    return show

def auto_close_polygon_if_near_first(points, scale, threshold_px=25):
    """Close polygon if the last vertex ended close to the first (display px threshold)."""
    if len(points) < 3: return points
    x0, y0 = points[0]
    xn, yn = points[-1]
    dx = (x0 - xn) * scale
    dy = (y0 - yn) * scale
    if (dx * dx + dy * dy) ** 0.5 <= threshold_px:
        points[-1] = points[0]
        if points[-1] != points[0]:
            points.append(points[0])
    return points

def parse_line(obj, scale):
    x1, y1 = obj.get("x1", 0), obj.get("y1", 0)
    x2, y2 = obj.get("x2", 0), obj.get("y2", 0)
    return [(x1 / scale, y1 / scale), (x2 / scale, y2 / scale)]

def parse_polygon(obj, scale):
    # prefer absolute 'path'
    if "path" in obj and isinstance(obj["path"], list):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M", "L"):
                pts.append((cmd[1] / scale, cmd[2] / scale))
        return pts
    # fallback to points + left/top + scale
    if "points" in obj and isinstance(obj["points"], list):
        left = obj.get("left", 0); top = obj.get("top", 0)
        sx = obj.get("scaleX", 1.0); sy = obj.get("scaleY", 1.0)
        pts = []
        for p in obj["points"]:
            px = left + sx * p.get("x", 0)
            py = top + sy * p.get("y", 0)
            pts.append((px / scale, py / scale))
        return pts
    return []

def safe_st_canvas(**kwargs):
    """
    Robust call to st_canvas across different package builds:
      1) try PIL.Image background
      2) try NumPy array background
      3) try explicit width/height + white background_color
    Also prefers display_ratio=1.0 to avoid DPI drift.
    """
    from streamlit_drawable_canvas import st_canvas
    bg = kwargs.pop("background_image")          # PIL.Image
    if bg.mode != "RGB":
        bg = bg.convert("RGB")
    W, H = bg.size
    bg_np = np.array(bg)

    def _call(**k):
        try:
            return st_canvas(display_ratio=1.0, **k)
        except TypeError:
            return st_canvas(**k)

    # A) PIL image (some builds accept PIL directly)
    try:
        return _call(background_image=bg, **kwargs)
    except Exception:
        pass

    # B) NumPy image (other builds require np.ndarray)
    try:
        return _call(background_image=bg_np, **kwargs)
    except Exception:
        pass

    # C) explicit size + white background (last resort)
    return _call(
        background_image=bg_np,
        width=W, height=H,
        background_color="#ffffff",
        **kwargs,
    )

# ============================ state ============================

ss = st.session_state
defaults = dict(
    poly=[], cora=None, hinge=None, prox=[], dist=[],
    dispw=1100, dx=0, dy=0, theta=0, segment="distal",
    tool_prev=None, canvas_nonce=0,
)
for k, v in defaults.items():
    ss.setdefault(k, v)

st.sidebar.header("Upload image")
uploaded = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg", "tif", "tiff"])

tool = st.sidebar.radio("Tool", ["Polygon", "CORA", "HINGE", "Prox line", "Dist line"], index=0)

# reset canvas (avoid ghost events) when switching tool
if ss.tool_prev != tool:
    ss.canvas_nonce += 1
    ss.tool_prev = tool

ss.segment = st.sidebar.radio(
    "Move segment", ["distal", "proximal"],
    index=(0 if ss.segment == "distal" else 1), horizontal=True
)
ss.dispw  = st.sidebar.slider("Preview width", 600, 1800, ss.dispw, 50)
ss.dx     = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy     = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta  = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)

c1, c2, c3, c4, c5 = st.sidebar.columns(5)
if c1.button("Undo"):
    if tool == "Polygon" and ss.poly:
        ss.poly.pop()
    elif tool == "Prox line":
        ss.prox.clear()
    elif tool == "Dist line":
        ss.dist.clear()
    elif tool == "CORA":
        ss.cora = None
    elif tool == "HINGE":
        ss.hinge = None
if c2.button("Reset poly"): ss.poly.clear()
if c3.button("Reset lines"): ss.prox.clear(); ss.dist.clear()
if c4.button("Clear centers"): ss.cora = None; ss.hinge = None
if c5.button("Reset move"): ss.dx = 0; ss.dy = 0; ss.theta = 0

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# ============================ image & scale ============================

base = decode_image(uploaded.getvalue())
W, H = base.size
scale = min(ss.dispw / float(W), 1.0)
disp_size = (int(round(W * scale)), int(round(H * scale)))
disp = base.resize(disp_size, Image.NEAREST)

# paint persisted shapes onto the canvas background so they stay visible
bg_for_canvas = draw_persisted_overlay(disp, 1.0, ss)

# ============================ canvas tools ============================

# Which drawing mode?
draw_mode = None
stroke_color = "#00FFFF"  # cyan polygon default
stroke_width = 2
if tool == "Polygon":
    # polygon if supported; polyline fallback still gives live preview
    draw_mode = "polygon"
elif tool == "Prox line":
    draw_mode = "line"; stroke_color = "#4285F4"; stroke_width = 3
elif tool == "Dist line":
    draw_mode = "line"; stroke_color = "#DD00DD"; stroke_width = 3
else:
    draw_mode = None  # CORA/HINGE handled by click on the image area (simple)

canvas_result = None
if draw_mode is not None:
    canvas_result = safe_st_canvas(
        background_image=bg_for_canvas,
        drawing_mode=draw_mode,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill_color="rgba(0,0,0,0)",
        update_streamlit=True,
        key=f"canvas-{ss.canvas_nonce}-{tool}",
    )
else:
    # If using CORA/HINGE, still show the image with persistence
    st.image(bg_for_canvas, use_column_width=False, width=disp_size[0])

# Parse what the user drew (if any)
if canvas_result is not None and canvas_result.json_data:
    objs = canvas_result.json_data.get("objects", [])
    # walk from last to first to get the most recent drawing
    for obj in reversed(objs):
        t = obj.get("type")
        if tool in ("Prox line", "Dist line") and t == "line":
            pts = parse_line(obj, scale)  # convert to original coords
            if tool == "Prox line":
                ss.prox = pts
            else:
                ss.dist = pts
            break
        if tool == "Polygon" and t in ("polygon", "polyline", "path"):
            pts = parse_polygon(obj, scale)
            if len(pts) >= 2:
                # If polygon tool but only polyline available, snap-close near first vertex
                pts = auto_close_polygon_if_near_first(pts, scale, threshold_px=25)
            if len(pts) >= 3:
                ss.poly = pts
            break

# CORA / HINGE placements (simple clicks on the static image under the canvas)
if tool in ("CORA", "HINGE"):
    # Use another minimal canvas to capture a single click reliably
    click_result = safe_st_canvas(
        background_image=bg_for_canvas,
        drawing_mode="transform",  # neutral mode; we only capture mouse up pos from JSON
        stroke_color="#00000000",
        stroke_width=0,
        update_streamlit=True,
        key=f"click-{ss.canvas_nonce}-{tool}",
    )
    # When user clicks, component creates a small JSON with current viewport;
    # we read pointer position from the frontend event (it’s placed as last path line).
    if click_result and click_result.json_data:
        # A tiny trick: when user clicks, a 'path' may appear with two identical points.
        objs = click_result.json_data.get("objects", [])
        if objs:
            # We can't reliably parse pointer here across versions;
            # So we mirror behavior: ask for last drawn point on polygon/line mode.
            # Fallback: do nothing if we didn't get coordinates.
            pass
    # Simpler + robust: add small radio-controlled manual entry
    with st.expander(f"Click to place {tool} (then optionally tweak)"):
        colx, coly = st.columns(2)
        x = colx.number_input("x (px)", value=float(ss.cora[0] if ss.cora and tool=="CORA" else (ss.hinge[0] if ss.hinge and tool=="HINGE" else disp_size[0]//2)), step=1.0)
        y = coly.number_input("y (px)", value=float(ss.cora[1] if ss.cora and tool=="CORA" else (ss.hinge[1] if ss.hinge and tool=="HINGE" else disp_size[1]//2)), step=1.0)
        # values are in display coords -> convert to original
        if st.button(f"Set {tool} here"):
            if tool == "CORA":
                ss.cora = (x/scale, y/scale)
            else:
                ss.hinge = (x/scale, y/scale)

# ============================ transform & output ============================

center = ss.hinge or ss.cora or centroid(ss.poly)
if len(ss.poly) >= 3 and center is not None:
    m = polygon_mask(base.size, ss.poly)
    inv = ImageOps.invert(m)
    prox_img = Image.new("RGBA", base.size, (0, 0, 0, 0)); prox_img.paste(base, (0, 0), inv)
    dist_img = Image.new("RGBA", base.size, (0, 0, 0, 0)); dist_img.paste(base, (0, 0), m)

    moving = dist_img if ss.segment == "distal" else prox_img
    fixed  = prox_img if ss.segment == "distal" else dist_img

    moved = apply_affine(moving, ss.dx, ss.dy, ss.theta, center)
    out = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", base.size, (0, 0, 0, 0)), fixed), moved)

    # redraw lines; each follows its own segment
    draw2 = ImageDraw.Draw(out)
    if len(ss.dist) == 2:
        p = transform_points_screen(ss.dist, ss.dx, ss.dy, ss.theta, center) if ss.segment == "distal" else ss.dist
        draw2.line(p, fill=(221, 0, 221, 255), width=3)
    if len(ss.prox) == 2:
        p = transform_points_screen(ss.prox, ss.dx, ss.dy, ss.theta, center) if ss.segment == "proximal" else ss.prox
        draw2.line(p, fill=(66, 133, 244, 255), width=3)

    disp_out = out.resize(disp_size, Image.NEAREST)
    st.image(disp_out, width=disp_size[0])

    # downloads
    params = dict(
        mode=ss.segment, dx=ss.dx, dy=ss.dy, rotate_deg=ss.theta,
        rotation_center=center, polygon_points=ss.poly,
        cora=ss.cora, hinge=ss.hinge, proximal_line=ss.prox, distal_line=ss.dist
    )
    df = pd.DataFrame([params])
    st.download_button("Download parameters CSV",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="osteotomy_params.csv", mime="text/csv", key="csv")
    buf = io.BytesIO(); out.save(buf, format="PNG")
    st.download_button("Download transformed image (PNG)",
                       data=buf.getvalue(), file_name="osteotomy_transformed.png",
                       mime="image/png", key="png")
else:
    st.info(
        "Canvas tips: draw polygon (it auto-closes when you finish near the first dot), "
        "then draw Prox/Dist lines. Switch tools from the sidebar; shapes persist. "
        "Set CORA/HINGE in the expander if needed."
    )
