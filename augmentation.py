import shutil
from PIL import ImageOps, ImageEnhance, Image, ImageFile
import os
import random


def do_base_augm(augmentation: str, dir_name_images, dir_name_textes):
    # Геометрическая аугментация всех файлов указанной директории
    all_imag = os.listdir(dir_name_images)

    for image_path in all_imag:

        image = Image.open(f'{dir_name_images}/{image_path}')
        example_text_path = f'{dir_name_textes}/{image_path[:-4]}.txt'


        ImageFile.LOAD_TRUNCATED_IMAGES=True

        with open(example_text_path) as old_text_file:
            old_information = list(old_text_file.read().split())
        new_information = old_information.copy()

        if augmentation == 'mirror':
            new_image = ImageOps.mirror(image)
            new_information[1] = str(1 - float(old_information[1]))
            new_information = ' '.join(new_information)

        if augmentation == 'flip':
            new_image = ImageOps.flip(image)
            new_information[2] = str(1 - float(old_information[2]))
            new_information = ' '.join(new_information)

        with open(example_text_path[:-4] + '_' + augmentation + '.txt', 'w') as new_text_file:
            new_text_file.write(new_information)

        new_image_name = image_path[:-4] + '_' + augmentation + '.jpg'
        new_image.save(f'{dir_name_images}/{new_image_name}')



def do_hard_augm(augmentation: str, dir_name_images, dir_name_textes):
    # Графическая аугментация всех файлов указанной директории
    all_imag = os.listdir(dir_name_images)

    for image_path in all_imag:
        image = Image.open(f'{dir_name_images}/{image_path}')
        example_text_path = f'{dir_name_textes}/{image_path[:-4]}.txt'

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(example_text_path) as old_text_file:
            old_information = old_text_file.read()

        if augmentation == 'sharpness':
            factors = [0.02, 0.5, 3, 5]
            enhancer = ImageEnhance.Sharpness(image)

        if augmentation == 'contrast':
            factors = [0.1, 0.5, 0.7, 3]
            enhancer = ImageEnhance.Contrast(image)

        i = 0
        for factor in factors:
            new_image = enhancer.enhance(factor)
            new_name = f'{dir_name_images}/{image_path[:-4]}_{augmentation}{str(i)}' + '.jpg'
            new_image.save(new_name)
            with open(f'{dir_name_textes}/{image_path[:-4]}_{augmentation}{str(i)}' + '.txt', 'w') as new_text_file:
                new_text_file.write(old_information)
            i += 1


def augmentate_it(dir_name_images: list[str], dir_name_textes: list[str]):
    # Директории с картинками и текстами к ним
    # dir_name_images = ['/content/peoples_labeled/train/images/', '/content/peoples_labeled/valid/images/']
    # dir_name_textes = ['/content/peoples_labeled/train/labels/', '/content/peoples_labeled/valid/labels/']

    for number in range(len(dir_name_images)):
        # Для каждого типа сорняка выполняются геометрическая и графическая аугментации
        # Их отличие в том, что при геометрической аугментации картинки необходим новый файл с описанием,
        # В то время как для графической просто копируется исходный

        list_of_base_augm = ['flip', 'mirror']
        for augmentation in list_of_base_augm:
            do_base_augm(augmentation, dir_name_images[number], dir_name_textes[number])

        list_of_hard_augm = ['sharpness', 'contrast']
        for augmentation in list_of_hard_augm:
            do_hard_augm(augmentation, dir_name_images[number], dir_name_textes[number])
