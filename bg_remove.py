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

def remove_background_from_image(input_path, output_folder):
    session = new_session()
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / (Path(input_path).stem + ".out.png"))

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, session=session)
            o.write(output)

    return output_path
    

# remove_background_from_folder('test-images', 'test-images-output')
print("test:")
print(remove_background_from_image('test-images/bg_white_top.png', 'test-images-output'))
print("end")
