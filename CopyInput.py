import shutil
import os

#shutil.copyfile(src, dst)


def main(input_folder):

    train_images = ".\\train\\images"
    train_masks = ".\\train\\masks"
    validation_images = ".\\validation\\images"
    validation_masks = ".\\validation\\masks"


    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_masks, exist_ok=True)
    os.makedirs(validation_images, exist_ok=True)
    os.makedirs(validation_masks, exist_ok=True)

    # Gather folders in train folder
    elements = os.listdir(f'{input_folder}\\train')

    for element in elements:
        image = os.listdir(f'{input_folder}\\train\\{element}\\images')[0]
        mask = os.listdir(f'{input_folder}\\train\\{element}\\masks')[0]

        shutil.copyfile(f'{input_folder}\\train\\{element}\\images\\{image}', f'{train_images}\\{image}')
        shutil.copyfile(f'{input_folder}\\train\\{element}\\masks\\{mask}', f'{train_masks}\\{mask}')


    # Gather folders in validation folder
    elements = os.listdir(f'{input_folder}\\test')

    for element in elements:
        image = os.listdir(f'{input_folder}\\test\\{element}\\images')[0]
        mask = os.listdir(f'{input_folder}\\test\\{element}\\masks')[0]

        shutil.copyfile(f'{input_folder}\\test\\{element}\\images\\{image}', f'{validation_images}\\{image}')
        shutil.copyfile(f'{input_folder}\\test\\{element}\\images\\{image}', f'{validation_masks}\\{mask}')

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    input_folder = "C:\\Users\\Pineirin\\Downloads\\input" 

    main(input_folder)