"""
Any photo a phone can send -> PNG bytes the rest of the pipeline can assume.

Four jobs, in this order, because each one has to happen before anything looks
at pixels. Everything downstream then works on one format at one known size.

Usage:
    python normalize.py photo.heic photo.png
"""
import io
import sys

import pillow_heif
from PIL import Image, ImageOps

MAX_EDGE = 1600   # above this, every contour bump grows its own skeleton twig

pillow_heif.register_heif_opener()   # so Image.open() accepts an iPhone .heic


def to_png(data: bytes) -> bytes:
    """Upright, RGB or RGBA, long edge at most MAX_EDGE."""
    image = Image.open(io.BytesIO(data))

    # A phone stores "rotate 90" as EXIF metadata and PIL does not apply it on
    # open. Skip this and the garment arrives sideways, the anchor step looks
    # for shoulders at the hem, and nothing downstream can tell.
    image = ImageOps.exif_transpose(image)

    # Palette and greyscale reach here from HEIC and PNG, and rembg and cv2
    # each fail differently on them.
    if image.mode not in ("RGB", "RGBA"):
        transparent = "A" in image.getbands() or "transparency" in image.info
        image = image.convert("RGBA" if transparent else "RGB")

    image.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)   # only ever shrinks

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python normalize.py photo.heic photo.png")

    with open(sys.argv[1], "rb") as f:
        png = to_png(f.read())
    with open(sys.argv[2], "wb") as f:
        f.write(png)
    print(f"{sys.argv[1]} -> {sys.argv[2]}, {len(png) / 1e6:.1f} MB")
