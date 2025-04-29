import albumentations as A
import cv2
from src.data import data_loader

import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),  # 90-degree random rotations
    A.RandomCrop(width=500, height=500, p=1.0),  # random crop
    A.Rotate(limit=15, p=0.5),  # Random rotation between -15 and +15 degrees
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.GaussianBlur(blur_limit=(1, 5), p=0.5),
    A.ISONoise(p=0.2),
],
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))





def augament_image_and_label(img_path: str):
    #
    # input args: img_path(str) -> This function takes an image path as input
    # return: augmented_image_object(nparray), adjusted_labels, original_image, original_boundingboxes
    # 
    # This function is used to generate augmented images and corresponding adjusted labels.
    # The image augmentation is randomly applied: rotation, crop, brightness, blur, noise.
    # The output image size is fixed to (500, 500, 3).
    #

    label_path = data_loader.get_image_label(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    class_labels = []   # [class_id]
    bboxes = []         # [[x_center, y_center, width, height]]
    full_labels = []    # [[class_id, x_center, y_center, width, height]]

    with open(label_path) as f:
        newline = f.readlines()
        for line in newline:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            class_labels.append(class_id)
            bboxes.append(bbox)
            full_labels.append([class_id] + bbox)

    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
    augmented_img = augmented['image']
    augmented_bboxes = augmented['bboxes']
    aug_class_labels = augmented['class_labels']

    augmented_labels = [[int(cls)] + list(bbox) for cls, bbox in zip(aug_class_labels, augmented_bboxes)]

    return augmented_img, augmented_labels, img, full_labels



