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

remove_background_from_folder('test-images', 'test-images-output')
