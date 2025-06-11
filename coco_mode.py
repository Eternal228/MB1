from file_working import rename_images, create_empty_dataset, fill_the_dataset, create_yaml, replace_first_number_in_files
from auto_annotating import annotate_it
from augmentation import augmentate_it

'''
input:
* images folder
* classes indexes

output:
* dataset
* * train
* * * images
* * * textes
* * valid
* * * images
* * * textes
* * yaml
'''


def coco_mode(images_folder:str, classes_indexes: list[int]):
    texts_folder = f'{images_folder}_texts'
    # rename_images(images_folder) # Переименование изображений
    annotate_it(input_dir=images_folder, output_dir=texts_folder)  # Создание текстовых файлов
    replace_first_number_in_files(texts_folder, classes_indexes)
    augmentate_it(dir_name_images=[images_folder], dir_name_textes=[texts_folder])
    create_empty_dataset()
    fill_the_dataset(images_folder, texts_folder)
    create_yaml('dataset', classes_indexes)


if __name__ == '__main__':
    images_folder = 'data'
    classes_indexes = [0, 27]
    coco_mode(images_folder, classes_indexes)
