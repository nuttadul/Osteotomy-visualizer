# simplify_bone_ninja.py
# Streamlit adapter expects this exact function signature.
# Replace internals with your own logic if desired.
from typing import List, Tuple
from PIL import Image, ImageOps, ImageDraw

def _polygon_mask(size: Tuple[int,int], points: List[Tuple[float,float]]) -> Image.Image:
    m = Image.new("L", size, 0)
    if len(points) >= 3:
        ImageDraw.Draw(m).polygon(points, fill=255, outline=255)
    return m

def _apply_affine(img: Image.Image, dx: float, dy: float, rot_deg: float, center_xy: Tuple[float,float]) -> Image.Image:
    img = img.convert("RGBA")
    rotated = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    canvas = Image.new("RGBA", img.size, (0,0,0,0))
    canvas.alpha_composite(rotated, (int(round(dx)), int(round(dy))))
    return canvas

def apply_transform(img_rgba: Image.Image,
                    polygon_pts: List[Tuple[float, float]],
                    center_xy: Tuple[float, float],
                    dx: float, dy: float, theta_deg: float,
                    segment: str = "distal") -> Image.Image:
    """
    Called by app.py
    img_rgba: PIL.Image (RGBA preferred)
    polygon_pts: osteotomy polygon (>=3 points)
    center_xy: rotation center
    dx, dy: translation in px
    theta_deg: rotation in degrees
    segment: "distal" or "proximal"
    """
    # Normalize orientation
    img = ImageOps.exif_transpose(img_rgba).convert("RGBA")

    # Build masks
    poly = _polygon_mask(img.size, polygon_pts)
    inv  = ImageOps.invert(poly)

    # Split into fragments
    prox = Image.new("RGBA", img.size, (0,0,0,0))
    dist = Image.new("RGBA", img.size, (0,0,0,0))
    prox.paste(img, (0,0), inv)
    dist.paste(img, (0,0), poly)

    move = dist if str(segment).lower() == "distal" else prox
    stay = prox if str(segment).lower() == "distal" else dist

    moved = _apply_affine(move, dx=dx, dy=dy, rot_deg=theta_deg, center_xy=center_xy)

    out = Image.alpha_composite(Image.new("RGBA", img.size, (0,0,0,0)), stay)
    out = Image.alpha_composite(out, moved)
    return out
