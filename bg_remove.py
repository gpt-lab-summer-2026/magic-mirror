import sys
from pathlib import Path
from rembg import remove, new_session

_session = None


def cutout(data: bytes) -> bytes:
    """Image bytes in, background-removed RGBA PNG bytes out."""
    global _session
    # Lazy: new_session() downloads ~170 MB of u2net on a fresh machine, so
    # building one at import time would hang even a --help.
    if _session is None:
        _session = new_session()
    return remove(data, session=_session)


def remove_background_from_folder(folder_path, output_folder):
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in Path(folder_path).glob('*.png'):
        remove_background_from_image(str(file), str(output_dir / (file.stem + ".out.png")))

def remove_background_from_image(input_path, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'rb') as i:
        data = i.read()
    with open(output_path, 'wb') as o:
        o.write(cutout(data))

    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python bg_remove.py path/to/input.png garments/name.png")

    print(remove_background_from_image(sys.argv[1], sys.argv[2]))
