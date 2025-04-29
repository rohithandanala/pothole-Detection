import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import data_loader
from src import predict
import torch
import yaml



def run_prediction():
    with open('configs/predict.yaml','r') as f:
        predict_configs = yaml.safe_load(f)


    img_list = data_loader.loadData(predict_configs['test_path'])
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=predict_configs['model_path'], force_reload=False)
    for i in img_list:
        print(f'predicting {i}')
        predict.run_yolov5_prediction(model, i, predict_configs['output_path'])

run_prediction()


