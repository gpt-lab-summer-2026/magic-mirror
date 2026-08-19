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

# What a camera roll actually holds. Pillow opens some thirty formats - FITS,
# GRIB, EPS - and each one is more C code for a malformed file to go after,
# none of it anything a phone would ever send.
FORMATS = {"PNG", "JPEG", "MPO", "HEIF", "WEBP"}

# Above any phone camera, below a bomb. Bytes say nothing about this: 10 MB of
# PNG holding one flat colour unpacks to gigabytes.
MAX_PIXELS = 50_000_000

pillow_heif.register_heif_opener()   # so Image.open() accepts an iPhone .heic


def to_png(data: bytes) -> bytes:
    """Upright, RGB or RGBA, long edge at most MAX_EDGE.

    ValueError carries a line to show whoever sent the file.
    """
    image = Image.open(io.BytesIO(data))

    # Both checks before anything below touches a pixel: Image.open reads the
    # header and stops, and it is the pixel decode that a crafted file aims at.
    if image.format not in FORMATS:
        raise ValueError(f"a {image.format} file is not a photo - send a PNG, JPEG or HEIC")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError(f"that photo is {image.width * image.height // 1_000_000} megapixels - too large")

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
