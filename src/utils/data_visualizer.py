import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_image_with_bboxes(image, bboxes, class_labels=None):
    """
    Args:
        image (np.array): Image array (500x500x3)
        bboxes (list): List of bounding boxes [x_center, y_center, width, height] (normalized between 0 and 1)
        class_labels (list, optional): List of class labels for each box
    """

    img_height, img_width = image.shape[:2]

    for idx, bbox in enumerate(bboxes):
        x_center, y_center, width, height = bbox[1], bbox[2], bbox[3], bbox[4] 

        # Convert normalized bbox to pixel values
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Get top-left and bottom-right corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

        # Draw class label if available
        if class_labels:
            label = str(class_labels[idx])
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show image using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
