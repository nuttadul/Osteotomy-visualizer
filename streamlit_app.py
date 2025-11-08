import io, math
from typing import List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

# live canvas
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Osteotomy – canvas over static background", layout="wide")

# ---------------- helpers ----------------
def decode_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def centroid(pts: List[Tuple[float, float]]):
    if len(pts) < 3:
        return None
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    a = 0.0; cx = 0.0; cy = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        cross = x[i]*y[j] - x[j]*y[i]
        a += cross
        cx += (x[i] + x[j]) * cross
        cy += (y[i] + y[j]) * cross
    a *= 0.5
    if abs(a) < 1e-9:
        return None
    cx /= (6*a); cy /= (6*a)
    return (cx, cy)

def polygon_mask(size, pts: List[Tuple[float,float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(pts) >= 3:
        ImageDraw.Draw(m).polygon(pts, fill=255, outline=255)
    return m

def apply_affine(img: Image.Image, dx, dy, rot_deg, center_xy):
    rot = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    out = Image.new("RGBA", img.size, (0,0,0,0))
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
        xr = x0*c + y0*s + cx + dx
        yr = -x0*s + y0*c + cy + dy
        out.append((float(xr), float(yr)))
    return out

def close_if_near_first(pts_disp: List[Tuple[float, float]], snap_px: float = 10.0):
    if len(pts_disp) < 3:
        return False, pts_disp
    x0,y0 = pts_disp[0]
    xn,yn = pts_disp[-1]
    if (x0-xn)**2 + (y0-yn)**2 <= snap_px**2:
        return True, pts_disp[:-1]
    return False, pts_disp

# ---- parse helpers (canvas → original-image coords) ----
def _line_points_from_obj(obj, scale):
    x1,y1 = obj.get("x1", 0), obj.get("y1", 0)
    x2,y2 = obj.get("x2", 0), obj.get("y2", 0)
    return [(x1/scale,y1/scale),(x2/scale,y2/scale)]

def _polygon_points_from_obj(obj, scale):
    # path (absolute M/L commands)
    if "path" in obj and isinstance(obj["path"], list):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M","L"):
                pts.append((cmd[1]/scale, cmd[2]/scale))
        return pts
    # fabric polygon points
    if "points" in obj and isinstance(obj["points"], list):
        left = obj.get("left", 0); top = obj.get("top", 0)
        sx = obj.get("scaleX", 1.0); sy = obj.get("scaleY", 1.0)
        pts = []
        for p in obj["points"]:
            px = left + sx * p.get("x", 0)
            py = top + sy * p.get("y", 0)
            pts.append((px/scale, py/scale))
        return pts
    return []

# ---------- SAFE canvas wrapper ----------
def safe_st_canvas(*, background_image, width, height, drawing_mode,
                   stroke_color, stroke_width, key):
    """
    Calls st_canvas but tolerates version differences:
      - Try with display_ratio=1.0 (+ numpy background)
      - Try without display_ratio
      - Try with PIL.Image background
      - Fallback to polyline if polygon mode unsupported
    """
    bg = background_image
    dm = drawing_mode

    def _call(**k):
        # most reliable combo first
        try:
            return st_canvas(display_ratio=1.0, **k)
        except TypeError:
            return st_canvas(**k)

    # 1) numpy + requested mode
    try:
        return _call(background_image=bg, width=width, height=height,
                     drawing_mode=dm, stroke_color=stroke_color,
                     stroke_width=stroke_width, key=key,
                     background_color="#ffffff", update_streamlit=True)
    except Exception:
        pass

    # 2) numpy + polyline (if polygon not supported)
    if dm == "polygon":
        try:
            return _call(background_image=bg, width=width, height=height,
                         drawing_mode="polyline", stroke_color=stroke_color,
                         stroke_width=stroke_width, key=key,
                         background_color="#ffffff", update_streamlit=True)
        except Exception:
            pass

    # 3) PIL background
    if isinstance(bg, np.ndarray):
        try:
            bg_pil = Image.fromarray(bg)
        except Exception:
            bg_pil = None
    else:
        bg_pil = bg

    if bg_pil is not None:
        # 3a) PIL + requested mode
        try:
            return _call(background_image=bg_pil, width=width, height=height,
                         drawing_mode=dm, stroke_color=stroke_color,
                         stroke_width=stroke_width, key=key,
                         background_color="#ffffff", update_streamlit=True)
        except Exception:
            pass
        # 3b) PIL + polyline
        if dm == "polygon":
            try:
                return _call(background_image=bg_pil, width=width, height=height,
                             drawing_mode="polyline", stroke_color=stroke_color,
                             stroke_width=stroke_width, key=key,
                             background_color="#ffffff", update_streamlit=True)
            except Exception:
                pass

    # final fallback: create a blank white background (should never be reached)
    blank = np.ones((height, width, 3), dtype=np.uint8) * 255
    return _call(background_image=blank, width=width, height=height,
                 drawing_mode="polyline" if dm == "polygon" else dm,
                 stroke_color=stroke_color, stroke_width=stroke_width,
                 key=key, background_color="#ffffff", update_streamlit=True)

# ---------- state ----------
ss = st.session_state
defaults = dict(
    poly=[], poly_closed=False,
    prox=[], dist=[],
    cora=None, hinge=None,
    segment="distal",
    dispw=1100, dx=0, dy=0, theta=0
)
for k,v in defaults.items():
    ss.setdefault(k, v)

# ---------- sidebar ----------
st.sidebar.header("Upload image")
up = st.sidebar.file_uploader(" ", type=["png","jpg","jpeg","tif","tiff"])

tool = st.sidebar.radio(
    "Tool",
    ["Polygon", "Prox line", "Dist line", "CORA", "HINGE"],
    index=0
)

ss.segment = st.sidebar.radio("Move segment", ["distal", "proximal"],
                              index=0 if ss.segment=="distal" else 1,
                              horizontal=True)

ss.dispw  = st.sidebar.slider("Preview width", 600, 1800, ss.dispw, 50)
ss.dx     = st.sidebar.slider("ΔX (px)", -1000, 1000, ss.dx, 1)
ss.dy     = st.sidebar.slider("ΔY (px)", -1000, 1000, ss.dy, 1)
ss.theta  = st.sidebar.slider("Rotate (°)", -180, 180, ss.theta, 1)

c1,c2,c3,c4,c5 = st.sidebar.columns(5)
if c1.button("Undo"):
    if tool=="Polygon":
        if ss.poly:
            ss.poly = ss.poly[:-1]
            if len(ss.poly)<3:
                ss.poly_closed=False
    elif tool=="Prox line" and ss.prox: ss.prox=[]
    elif tool=="Dist line" and ss.dist: ss.dist=[]
    elif tool=="CORA": ss.cora=None
    elif tool=="HINGE": ss.hinge=None
if c2.button("Reset poly"): ss.poly=[]; ss.poly_closed=False
if c3.button("Reset lines"): ss.prox=[]; ss.dist=[]
if c4.button("Clear centers"): ss.cora=None; ss.hinge=None
if c5.button("Reset move"): ss.dx=0; ss.dy=0; ss.theta=0

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# ---------- base image & scale ----------
img_rgba = decode_image(up.getvalue())
W,H = img_rgba.size
scale = min(ss.dispw/float(W), 1.0)
disp_size = (int(round(W*scale)), int(round(H*scale)))
disp_rgb = img_rgba.convert("RGB").resize(disp_size, Image.NEAREST)
bg_np = np.array(disp_rgb)   # static background

# picking drawing mode
drawing_mode = {
    "Polygon":   "polygon",       # wrapper will fallback to 'polyline' if needed
    "Prox line": "line",
    "Dist line": "line",
    "CORA":      None,            # no drawing, just clicks
    "HINGE":     None,
}[tool]

# ---------- LIVE CANVAS (safe) ----------
result = safe_st_canvas(
    background_image=bg_np,
    width=disp_size[0],
    height=disp_size[1],
    drawing_mode=drawing_mode,
    stroke_color="#00FFFF" if tool=="Polygon" else "#4285F4" if tool=="Prox line" else "#DD00DD",
    stroke_width=3,
    key="main-canvas",
)

# ---------- read shapes & update state ----------
def _to_orig(pt_disp): return (pt_disp[0]/scale, pt_disp[1]/scale)

if result and result.json_data:
    objs = result.json_data.get("objects", [])

    # polygon / polyline / path
    for obj in reversed(objs):
        if obj.get("type") in ("polygon","polyline","path"):
            if obj["type"]=="polygon":
                left = obj.get("left",0); top=obj.get("top",0)
                sx = obj.get("scaleX",1.0); sy=obj.get("scaleY",1.0)
                pts_disp=[]
                for p in obj.get("points", []):
                    pts_disp.append((left + sx*p.get("x",0), top + sy*p.get("y",0)))
            else:
                pts_disp=[]
                for cmd in obj.get("path", []):
                    if isinstance(cmd,list) and len(cmd)>=3 and cmd[0] in ("M","L"):
                        pts_disp.append((cmd[1], cmd[2]))
            if pts_disp:
                closed, pts_disp = close_if_near_first(pts_disp, snap_px=10.0)
                ss.poly = [ _to_orig(p) for p in pts_disp ]
                ss.poly_closed = closed
            break

    # lines
    for obj in reversed(objs):
        if obj.get("type") == "line":
            p_disp = [(obj.get("x1",0), obj.get("y1",0)),
                      (obj.get("x2",0), obj.get("y2",0))]
            p = [ _to_orig(pt) for pt in p_disp ]
            if tool == "Prox line":
                ss.prox = p
            elif tool == "Dist line":
                ss.dist = p
            break

    # CORA / HINGE click
    last_pt = getattr(result, "last_point", None)
    if tool in ("CORA","HINGE") and last_pt is not None:
        px,py = last_pt["x"], last_pt["y"]
        if tool=="CORA": ss.cora = _to_orig((px,py))
        else:            ss.hinge = _to_orig((px,py))

# ---------- overlay preview ----------
overlay = disp_rgb.copy()
d = ImageDraw.Draw(overlay)

if ss.poly:
    poly_disp = [(p[0]*scale, p[1]*scale) for p in ss.poly]
    if len(poly_disp)>=2:
        d.line(poly_disp, fill=(0,255,255), width=2)
    if ss.poly_closed and len(poly_disp)>=3:
        d.line([*poly_disp, poly_disp[0]], fill=(0,255,255), width=2)
    for x,y in poly_disp:
        d.ellipse([x-3,y-3,x+3,y+3], fill=(0,255,255))
if len(ss.prox)==2:
    a,b = ss.prox; d.line([(a[0]*scale,a[1]*scale),(b[0]*scale,b[1]*scale)], fill=(66,133,244), width=3)
if len(ss.dist)==2:
    a,b = ss.dist; d.line([(a[0]*scale,a[1]*scale),(b[0]*scale,b[1]*scale)], fill=(221,0,221), width=3)
if ss.cora:
    x,y = ss.cora; x*=scale; y*=scale
    d.ellipse([x-6,y-6,x+6,y+6], outline=(0,200,0), width=2)
if ss.hinge:
    x,y = ss.hinge; x*=scale; y*=scale
    d.ellipse([x-7,y-7,x+7,y+7], outline=(255,165,0), width=3)
    d.line([(x-10,y),(x+10,y)], fill=(255,165,0), width=1)
    d.line([(x,y-10),(x,y+10)], fill=(255,165,0), width=1)

st.image(overlay, caption="Live overlay on static background", use_container_width=False)

# ---------- backend transform + download ----------
center = ss.hinge or ss.cora or (centroid(ss.poly) if ss.poly_closed else None)

if ss.poly_closed and center is not None:
    img = img_rgba
    m = polygon_mask(img.size, ss.poly)
    inv = ImageOps.invert(m)
    prox_img = Image.new("RGBA", img.size, (0,0,0,0)); prox_img.paste(img, (0,0), inv)
    dist_img = Image.new("RGBA", img.size, (0,0,0,0)); dist_img.paste(img, (0,0), m)
    moving = dist_img if ss.segment=="distal" else prox_img
    fixed  = prox_img if ss.segment=="distal" else dist_img
    moved  = apply_affine(moving, ss.dx, ss.dy, ss.theta, center)
    out    = Image.alpha_composite(Image.alpha_composite(Image.new("RGBA", img.size, (0,0,0,0)), fixed), moved)

    # redraw lines according to which segment moves
    draw2 = ImageDraw.Draw(out)
    if len(ss.dist)==2:
        p = transform_points_screen(ss.dist, ss.dx, ss.dy, ss.theta, center) if ss.segment=="distal" else ss.dist
        draw2.line(p, fill=(221,0,221,255), width=3)
    if len(ss.prox)==2:
        p = transform_points_screen(ss.prox, ss.dx, ss.dy, ss.theta, center) if ss.segment=="proximal" else ss.prox
        draw2.line(p, fill=(66,133,244,255), width=3)

    disp_out = out.resize(disp_size, Image.NEAREST)
    st.image(disp_out, caption="Transformed preview", use_container_width=False)

    params = dict(
        mode=ss.segment, dx=ss.dx, dy=ss.dy, rotate_deg=ss.theta,
        rotation_center=center, polygon_points=ss.poly,
        cora=ss.cora, hinge=ss.hinge,
        proximal_line=ss.prox, distal_line=ss.dist
    )
    df = pd.DataFrame([params])
    st.download_button(
        "Download parameters CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="osteotomy_params.csv",
        mime="text/csv",
        key="csv"
    )
    buf = io.BytesIO(); out.save(buf, format="PNG")
    st.download_button(
        "Download transformed image (PNG)",
        data=buf.getvalue(),
        file_name="osteotomy_transformed.png",
        mime="image/png",
        key="png"
    )
else:
    st.info("Draw polygon. To close it, click near the first vertex (auto-close). Then set CORA/HINGE and move a segment.")
