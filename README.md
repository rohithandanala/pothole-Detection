# 🕳️ Pothole Detection using YOLOv5

This project uses **YOLOv5** to detect potholes in road images. It supports training custom models, performing predictions, and visualizing results. The model is trained on a custom dataset and uses PyTorch for inference.
![Sample Output](assets/potholes.gif)

---

## ✅ Features

- 🎯 YOLOv5-based Object Detection Custom-trained YOLOv5 model to accurately detect potholes in road images.
- 🏋️ Custom Model Training Support Easily train your own model on labeled datasets using configurable training scripts.
- 📁 Batch Image Prediction Run inference on a folder of images and automatically save predictions to a target directory.
- 🖼️ Tkinter GUI for Predictions Simple, user-friendly interface with drag-and-drop image upload and grid-style result viewing.
- 🔄 No Redundant Downloads Optimized to avoid re-downloading the YOLOv5 repo or weights on repeated runs.
- 🗂️ Organized Output Structure Predicted images are saved neatly in a single folder without unnecessary subdirectories.
- 🧩 Modular Codebase Clear separation between training, prediction, GUI, and utility functions for easy maintenance.
- 🔭 Scalable for Future Upgrades Designed with future integration in mind: MLflow for tracking and AWS for automated retraining pipelines.

---

## 🏗️ Project Structure

```
├── UI/
├── assets/
├── configs/
│   └── data.yaml             # Training configuration file
│   └── predict.yaml             # Prediction configuration file
├── data/
│   └── raw/                    # Raw
│   │   ├── Images              # raw images
│   └── └── Labels              # Raw labels 
├── outputs/
│   └── runs/                  # saved models
├── pipelines/
│   ├── run_pipeline.py         # Main pipeline script
│   └── predict_potholes.py     # prediction script
├── predictions/
│   ├── Inputs/         #  <-----input images goes here
│   └── Outputs         #  <-----Predicted images saved here
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Load datasets
│   │   └── data_augamentor.py   # augaments data
│   ├── models/
│   ├── utils/
│   │  ├── check_gpu.py            # check if gpu is available
│   │  ├── data_visualizer.py           # visualize image with bounding boxes
│   │  ├── generate_empty_dir.py        # Generates empty directories for training data
│   │  ├── mlflow_train.py              # mlflow for model monitoring
│   │  ├── save_image_and_label.py      # save images and labels at desired location
│   │  └── train_test_splitter.py       # split given raw data into test train in given ratio
│   ├── predict.py    # predict objects in a given image
│   ├── train.py      # trains model with given settings
├── tests/    '
│   └── code_testing.py     #script to test function performance              
├── requirements.txt            # Project dependencies
├── README.md
└── .gitignore
```

---

## 🚀 Quickstart Guide

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
[git clone https://github.com/your-username/pothole-detection-yolov5.git](https://github.com/rohithandanala/pothole-Detection.git)
cd pothole-detection-yolov5
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
# Activate the environment
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
```
For GPU usage
```bash
python -m venv venv
# Activate the environment
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
nvidia-smi    #check if compatable GPU available in your device. if doesn't work please download cuda in link provided below
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   #Download suitable version
```
- Download cuda at here [cuda](https://developer.nvidia.com/cuda-zone)
- Download torch (GPU version) at here. please choose suitable version [torch-gpu](https://pytorch.org/get-started/locally/)


### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5  # to download required script for yolo
```

### 4. Prepare Your Data

- Place your data images **Images** and **Labels** inside the `data/raw/` folder.
- Ensure your labels should be in yolo format i.e., <class> <x_center> <y_center> <height> <width>.

Example structure:

```
[0 0.2 0.3 0.1 0.4]
...
```

- Edit the configuration file `configs/data.yaml` to match your data paths and training settings:

```yaml
python yolov5/train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data configs/data.yaml \
  --weights yolov5s.pt \
  --name pothole_yolov5s \
  --device 0 \
  --project outputs/runs \
  --exist-ok
```

### 5. Run the Full Training Pipeline

```bash
python pipelines/run_pipeline.py --for training
python pipeline/predict_potholes.py --for prediction
```

✅ This will:

- Train the yolo model
- saves weights in `outputs/runs/`

---


## 📊 Prediction

Each time you run training:

- The trained model weights is logged inside the `outputs/runs/model_name/best.pt` directory.
- change this directory in predict.yaml to change model.
- place all images needed to predict in `predictions/Input`
- predicted output can be seen in `predictions/Output`


---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome!\
Feel free to ⭐ the repo if you find it useful!

---

## 📬 Contact

Created by Venkata Sai Rohith Andanala – feel free to reach out via GitHub Issues or pull requests.

---
