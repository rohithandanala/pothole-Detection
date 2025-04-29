import torch

def run_yolov5_prediction(weights_path, source_path, output_dir='runs/predict'):
    """
    Run YOLOv5 prediction on images.

    Args:
        weights_path (str): Path to trained .pt file (e.g., 'runs/train/exp/weights/best.pt')
        source_path (str): Path to image or directory of images
        output_dir (str): Directory to save prediction results
    """
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

    # Run inference
    results = model(source_path)

    # Save results
    results.save(save_dir=output_dir)

    print(f"✅ Predictions saved to: {output_dir}")

