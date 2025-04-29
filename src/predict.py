import cv2
import os

def run_yolov5_prediction(model, source_path, output_dir='runs/predict'):
    """
    Run YOLOv5 prediction and save results without creating subfolders.

    Args:
        model: YOLOv5 model loaded via torch.hub
        source_path (str or list): Path(s) to image(s)
        output_dir (str): Directory to save prediction results
    """
    # Run inference
    results = model(source_path)

    # Render predictions
    results.render()  # Populates `results.ims` with annotated images (np arrays)

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each annotated image
    for i, img in enumerate(results.ims):  # Use `ims`, not `imgs`
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = os.path.basename(results.files[i])
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img_bgr)

    print(f"✅ Predictions saved to: {output_dir}")

