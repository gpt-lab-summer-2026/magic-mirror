import sys
from pathlib import Path
from rembg import remove, new_session

def remove_background_from_folder(folder_path, output_folder):
    session = new_session()
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in Path(folder_path).glob('*.png'):
        input_path = str(file)
        output_path = str(output_dir / (file.stem + ".out.png"))

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)

def remove_background_from_image(input_path, output_path):
    session = new_session()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, session=session)
            o.write(output)

    return output_path


# remove_background_from_folder('test-images', 'test-images-output')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python bg_remove.py path/to/input.png garments/name.png")

    print(remove_background_from_image(sys.argv[1], sys.argv[2]))
