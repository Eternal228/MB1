import shutil

import cv2
import numpy as np
from PIL import ImageOps, ImageEnhance, Image, ImageFile
import os
import random

from ultralytics.data.augment import RandomHSV


def do_base_augm(augmentation: str, dir_name_images, dir_name_textes):
    # Геометрическая аугментация всех файлов указанной директории
    all_imag = os.listdir(dir_name_images)

    for image_path in all_imag:

        image = Image.open(f'{dir_name_images}/{image_path}')
        if dir_name_images:
            example_text_path = f'{dir_name_textes}/{image_path.split('.')[-2]}.txt'

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if dir_name_textes:
            with open(example_text_path) as old_text_file:
                old_information = [i.split() for i in old_text_file.readlines()]

        new_information = []
        if augmentation == 'mirror':
            new_image = ImageOps.mirror(image)
            if dir_name_textes:
                for old_line in old_information:
                    new_line = old_line.copy()
                    for i in range(1, len(old_line), 2):
                        new_line[i] = str(1 - float(old_line[i]))
                    new_information.append(' '.join(new_line))

        if augmentation == 'flip':
            new_image = ImageOps.flip(image)
            if dir_name_textes:
                for old_line in old_information:
                    new_line = old_line.copy()
                    for i in range(2, len(old_line), 2):
                        new_line[i] = str(1 - float(old_line[i]))
                    new_information.append(' '.join(new_line))

        new_information = '\n'.join(new_information)
        if dir_name_textes:
            with open(example_text_path[:-4] + '_' + augmentation + '.txt', 'w') as new_text_file:
                new_text_file.write(new_information)

        new_image_name = image_path.split('.')[-2] + '.' + image_path.split('.')[-1] + '_' + augmentation + '.jpg'
        new_image.save(f'{dir_name_images}/{new_image_name}')


def do_hard_augm(augmentation: str, dir_name_images, dir_name_textes):
    # Графическая аугментация всех файлов указанной директории

    image_extensions = ('.jpg', '.jpeg', '.png')
    all_imag = [i for i in os.listdir(dir_name_images) if i.lower().endswith(image_extensions)]
    for image_path in all_imag:

        image = Image.open(f'{dir_name_images}/{image_path}')
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if dir_name_textes:
            example_text_path = f'{dir_name_textes}/{image_path.split('.')[-2]}.txt'
            with open(example_text_path) as old_text_file:
                old_information = old_text_file.read()

        if augmentation == 'sharpness':
            factors = [0.5, 5]
            enhancer = ImageEnhance.Sharpness(image)

        if augmentation == 'contrast':
            factors = [0.4, 3]
            enhancer = ImageEnhance.Contrast(image)

        i = 0
        for factor in factors:
            new_image = enhancer.enhance(factor)
            new_name = f'{dir_name_images}/{image_path.split('.')[-2]}_{augmentation}{str(i)}.{image_path.split('.')[-1]}'
            new_image.save(new_name)
            if dir_name_textes:
                with open(f'{dir_name_textes}/{image_path.split('.')[-2]}_{augmentation}{str(i)}' + '.txt', 'w') as new_text_file:
                    new_text_file.write(old_information)
            i += 1


def do_hsv_augm(dir_name_images, dir_name_textes):
    # Список кортежей аугметнации типа
    # (Цветовая гамма, насыщенность, яркость)
    factors = [
        (0.7, 1.0, 0.5),
        (0.9, 0.2, 0.3),  # Сдвиг в сторону фиолетового, низкая насыщенность и яркость
        (0.3, 0.7, 0.6),  # Сдвиг в сторону зеленого, средняя насыщенность и яркость
        (0.9, 0.5, 1.0),  # Сдвиг в сторону синего, низкая насыщенность, высокая яркость
        (0.5, 0.0, 0.8)  # Без изменения цветового тона, обесцвечивание, максимальная яркость
    ]

    all_imag = os.listdir(dir_name_images)
    for image_path in all_imag:
        image_name = f'{dir_name_images}/{image_path}'
        image = cv2.imread(image_name)
        example_text_path = f'{dir_name_textes}/{image_path.split('.')[-2]}.txt'
        with open(example_text_path) as old_text_file:
            old_information = old_text_file.read()

        i = 0
        for factor in factors:
            new_image_name = f'{dir_name_images}/{image_path.split('.')[-2]}_hsv{str(i)}.{image_path.split('.')[-1]}'

            # 3. Применение HSV-аугментации
            augmented_image = augment_hsv(image.copy(), *factor)  # Важно использовать copy(), чтобы не менять исходное изображение

            '''
            # 4. Отображение оригинального и аугментированного изображений (опционально)
            cv2.imshow('Original Image', image)
            cv2.imshow('Augmented Image', augmented_image)
            cv2.waitKey(0)  # Ждем нажатия клавиши, чтобы закрыть окна
            cv2.destroyAllWindows()
            '''

            # 5. Сохранение аугментированного изображения (опционально)
            cv2.imwrite(new_image_name, augmented_image)

            with open(f'{example_text_path.split('.')[-2]}_hsv{str(i)}.txt', 'w') as new_text_file:
                new_text_file.write(old_information)
            i += 1


def do_yolohsv_augm(dir_name_images, dir_name_textes):
    # Список кортежей аугметнации типа
    # (Цветовая гамма, насыщенность, яркость)
    factors = [
        (0.9, 0.2, 0.3),  # Сдвиг в сторону фиолетового, низкая насыщенность и яркость
        (0.3, 0.7, 0.6),  # Сдвиг в сторону зеленого, средняя насыщенность и яркость
        (0.7, 0.4, 0.8),  # Сдвиг в сторону синего, низкая насыщенность, высокая яркость
        (0.5, 0.0, 1.0),  # Без изменения цветового тона, обесцвечивание, максимальная яркость
        (0.5, 1.0, 0.0),  # Без изменения цветового тона, максимальная насыщенность, полная темнота
    ]
    '''
    factors = [
        (0.7, 1.0, 0.5),
        (0.9, 0.2, 0.3),  # Сдвиг в сторону фиолетового, низкая насыщенность и яркость
        (0.3, 0.7, 0.6),  # Сдвиг в сторону зеленого, средняя насыщенность и яркость
        (0.9, 0.5, 1.0),  # Сдвиг в сторону синего, низкая насыщенность, высокая яркость
        (0.5, 0.0, 0.8)  # Без изменения цветового тона, обесцвечивание, максимальная яркость
    ]
    '''

    image_extensions = ('.jpg', '.jpeg', '.png')
    all_imag = [i for i in os.listdir(dir_name_images) if i.lower().endswith(image_extensions)]
    for image_path in all_imag:

        image_name = f'{dir_name_images}/{image_path}'

        image = cv2.imread(image_name)
        if dir_name_textes:
            example_text_path = f'{dir_name_textes}/{image_path.split('.')[-2]}.txt'
            with open(example_text_path) as old_text_file:
                old_information = old_text_file.read()

        i = 0
        for factor in factors:
            new_image_name = f'{image_name.split('.')[-2]}_yolohsv{str(i)}.{image_path.split('.')[-1]}'
            hsv_augm = RandomHSV(*factor)
            labels = {'img': image}
            hsv_augm(labels)
            augmented_image = labels["img"]
            # 3. Применение HSV-аугментации
            # Важно использовать copy(), чтобы не менять исходное изображение

            '''
            # 4. Отображение оригинального и аугментированного изображений (опционально)
            cv2.imshow('Original Image', image)
            cv2.imshow('Augmented Image', augmented_image)
            cv2.waitKey(0)  # Ждем нажатия клавиши, чтобы закрыть окна
            cv2.destroyAllWindows()
            '''

            # 5. Сохранение аугментированного изображения (опционально)
            cv2.imwrite(new_image_name, augmented_image)
            if dir_name_textes:
                with open(f'{example_text_path.split('.')[-2]}_yolohsv{str(i)}.txt', 'w') as new_text_file:
                    new_text_file.write(old_information)
                i += 1


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    Применяет HSV-аугментацию к изображению.

    Аргументы:
        img (numpy.ndarray): Исходное изображение (в формате BGR).
        hgain (float): Коэффициент изменения оттенка (Hue).
        sgain (float): Коэффициент изменения насыщенности (Saturation).
        vgain (float): Коэффициент изменения яркости (Value).

    Возвращает:
        numpy.ndarray: Аугментированное изображение.
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # сохраняем тип данных

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    return img


def augmentate_it(dir_name_images: str, dir_name_textes=None):
    # Директории с картинками и текстами к ним
    # Если dir_name_textes = None, выполняется аугментация без разметки текстовых файлов


    list_of_base_augm = ['flip', 'mirror']
    for augmentation in list_of_base_augm:
        do_base_augm(augmentation, dir_name_images, dir_name_textes)

    list_of_hard_augm = ['contrast']
    for augmentation in list_of_hard_augm:
        do_hard_augm(augmentation, dir_name_images, dir_name_textes)

    do_yolohsv_augm(dir_name_images, dir_name_textes)
    #do_hsv_augm(dir_name_images, dir_name_textes)
