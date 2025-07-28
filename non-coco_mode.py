import numpy as np
import torch
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import os
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

from file_working import create_empty_dataset, fill_the_dataset, create_yaml
from augmentation import augmentate_it


def zero_shot_folder_detection(image_dir: str, classes: list[str], output_dir=None, min_confidence: float = 0.03,
                               model_id: str = "yolo_world/l"):
    output_dir = output_dir or f'{image_dir}_texts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = YOLOWorld(model_id=model_id)
    model.set_classes(classes)

    # Список расширений изображений
    image_extensions = ('.jpg', '.jpeg', '.png')

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(image_dir, filename)
            zero_shot_image_detection(file_path, model, output_dir, min_confidence)


def zero_shot_image_detection(image_path: str, model, output_dir, min_confidence: float = 0.03):
    image = cv2.imread(image_path)
    results = model.infer(image, confidence=min_confidence)
    detections = sv.Detections.from_inference(results)
    print(detections.xyxy.tolist())
    hints = detections.xyxy.tolist()
    x, y, _ = image.shape
    print(x, y)
    for i in range(len(hints)):
        hints[i][0] /= x
        hints[i][2] /= x
        hints[i][1] /= y
        hints[i][3] /= y
    info = [
        f"{classid} {' '.join([str(i) for i in hint])}\n"
        for classid, hint
        in zip(detections.class_id, hints)
    ]
    image_path = image_path.split('\\')[-1]
    text_filename = f"{output_dir}/{image_path.replace(os.path.splitext(image_path)[1], ".txt")}"
    with open(text_filename, 'w') as f:
        f.writelines(info)


def sam_folder_segmentation(image_dir: str, texts_dir: str, sam_checkpoint="sam_vit_l_0b3195.pth", model_type="vit_l"):
    sys.path.append("..")
    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'
        print('cuda is active')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    image_extensions = ('.jpg', '.jpeg', '.png')
    all_images_paths = [
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(image_extensions)
    ]
    for image_path in all_images_paths:
        sam_image_segmentation(predictor, image_path, texts_dir)


def sam_image_segmentation(predictor, image_path: str, texts_dir: str):
    image_base_path = image_path.split('.')
    image_base_path[-1] = 'txt'
    image_base_path = ('.'.join(image_base_path)).split('\\')[-1]
    text_filename = fr"{texts_dir}\{image_base_path}"

    # text_filename = f"{texts_dir}/{image_path.replace(os.path.splitext(image_path)[1], ".txt")}"

    with open(text_filename, 'r') as f:
        lines = f.readlines()
    lines = [(line.rstrip()).split() for line in lines]
    '''
    for line in lines:
        class_id = int(line[0])
        hints = [int(i) for i in line[1:]]
    '''
    bboxes = [(int(line[0]), [float(i) for i in line[1:]]) for line in lines]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    x, y, _ = image.shape
    information = ''
    for bbox in bboxes:
        class_number, xyxy = bbox
        xyxy[0] *= x
        xyxy[1] *= y
        xyxy[2] *= x
        xyxy[3] *= y
        input_box = np.array(xyxy)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )

        h, w = masks.shape[1], masks.shape[2]

        # Объединим все маски в одну с помощью логического ИЛИ
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            combined_mask = cv2.bitwise_or(combined_mask, m.astype(np.uint8))

        # Находим контуры объединённой маски
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            print("Контуры не найдены")
            continue

        # Выбираем контур с максимальным числом точек (главный контур)
        contour = max(contours, key=len)
        contour_coords = contour[:, 0, :]  # shape: (N, 2)

        # Преобразование в относительные координаты (x/w, y/h)
        contour_coords_rel = contour_coords.astype(np.float32)
        contour_coords_rel[:, 0] /= w  # нормализация по ширине
        contour_coords_rel[:, 1] /= h  # нормализация по высоте

        # Преобразуем в "плоский" список [x1, y1, x2, y2, ...]
        contour_coords_rel_list = contour_coords_rel.flatten().tolist()

        information += f"{class_number} {' '.join([str(i) for i in contour_coords_rel_list])}\n"

    with open(text_filename, 'w') as f:
        f.writelines(information)


def non_coco_mode(images_folder: str, classes: list[str]):
    texts_folder = f'{images_folder}_texts'
    zero_shot_folder_detection(images_folder, classes, texts_folder, min_confidence=0.025)
    sam_folder_segmentation(images_folder, texts_folder)
    print('segmentation is ready')
    augmentate_it(dir_name_images=[images_folder], dir_name_textes=[texts_folder])
    print('augmentation is ready')
    dataset_name = 'dataset_sam'
    create_empty_dataset(dataset_name)
    fill_the_dataset(dataset_name, images_folder, texts_folder)
    create_yaml(dataset_name, list(range(len(classes))), classes)


if __name__ == "__main__":
    images_folder = 'data'
    classes = ['man', 'head', 'hand', 'eye', 'nose', 'apple', 'pants']
    non_coco_mode(images_folder, classes)
