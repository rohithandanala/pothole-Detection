import mlflow
import os
import shutil

# Start MLflow run
with mlflow.start_run(run_name="YOLOv5_Pothole_Training"):

    # Log training config
    mlflow.log_params({
        "image_size": data_configs['n_img_size'],
        "batch_size": data_configs['n_batch'],
        "epochs": data_configs['n_epochs'],
        "model": "yolov5s"
    })

    # Run training
    os.system(
        f"python yolov5/train.py "
        f"--img {data_configs['n_img_size']} "
        f"--batch {data_configs['n_batch']} "
        f"--epochs {data_configs['n_epochs']} "
        f"--data configs/data.yaml "
        f"--weights yolov5s.pt "
        f"--name exp1 "
        f"--project {custom_model_dir} "
        f"--device 0 "
        f"--workers 0"
    )

    # Path to saved best model
    best_model_path = os.path.join(custom_model_dir, "exp1", "weights", "best.pt")

    # Log model artifact
    mlflow.log_artifact(best_model_path, artifact_path="models")

    print("✅ Training complete and model logged to MLflow.")
