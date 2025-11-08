"""
engine.py
A minimal, robust transformation engine for the Streamlit adapter.

Implement `apply_transform` to cut along a polygon, choose a rotation center,
and move either the distal or proximal fragment by (dx, dy, theta_deg).

Return value must be a PIL.Image in RGBA.
"""

from typing import List, Tuple
from PIL import Image, ImageDraw, ImageOps


def _polygon_mask(size: Tuple[int, int], points: List[Tuple[float, float]]) -> Image.Image:
    """Create a binary mask (L mode) for the polygon."""
    mask = Image.new("L", size, 0)
    if points and len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=255, outline=255)
    return mask


def _apply_affine(img: Image.Image, dx: float, dy: float, rot_deg: float, center_xy: Tuple[float, float]) -> Image.Image:
    """Rotate around center, then translate (dx, dy). Keeps canvas size; no expansion."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    rotated = img.rotate(rot_deg, resample=Image.BICUBIC, center=center_xy, expand=False)
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    # round to nearest pixel to avoid seams; feel free to change to int(dx), int(dy)
    canvas.alpha_composite(rotated, (int(round(dx)), int(round(dy))))
    return canvas


def _composite(base: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
    """Paste overlay onto a transparent canvas using mask, then alpha-composite onto base."""
    out = base.copy()
    out.paste(overlay, (0, 0), mask)
    return out


def apply_transform(
    img_rgba: Image.Image,
    polygon_pts: List[Tuple[float, float]],
    center_xy: Tuple[float, float],
    dx: float,
    dy: float,
    theta_deg: float,
    segment: str = "distal",
) -> Image.Image:
    """
    Core engine API used by the Streamlit adapter.

    Parameters
    ----------
    img_rgba : PIL.Image (RGBA preferred)
        Source image.
    polygon_pts : list[(x, y)]
        Osteotomy polygon (>= 3 points). Inside is the distal fragment by convention.
    center_xy : (cx, cy)
        Rotation center.
    dx, dy : float
        Translation in pixels.
    theta_deg : float
        Rotation in degrees.
    segment : str
        "distal" to move the polygon area, "proximal" to move the outside.

    Returns
    -------
    PIL.Image (RGBA)
        Transformed composite image.
    """
    # Normalize mode/orientation
    img = ImageOps.exif_transpose(img_rgba).convert("RGBA")

    # Create masks
    poly_mask = _polygon_mask(img.size, polygon_pts)
    inv_mask = ImageOps.invert(poly_mask)

    # Split into proximal and distal layers
    transparent = Image.new("RGBA", img.size, (0, 0, 0, 0))
    proximal = transparent.copy()
    distal = transparent.copy()
    proximal.paste(img, (0, 0), inv_mask)  # outside polygon
    distal.paste(img, (0, 0), poly_mask)   # inside polygon

    # Choose which fragment to move
    moving = distal if (str(segment).lower() == "distal") else proximal
    fixed   = proximal if (str(segment).lower() == "distal") else distal

    # Transform the moving fragment
    moved = _apply_affine(moving, dx=dx, dy=dy, rot_deg=theta_deg, center_xy=center_xy)

    # Composite back together
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out = Image.alpha_composite(canvas, fixed)
    out = Image.alpha_composite(out, moved)

    return out
