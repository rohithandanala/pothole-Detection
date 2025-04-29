import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import data_loader
from src import predict

def run_prediction():
    img_list = data_loader.loadData('data/test')
    for i in img_list:
        print(f'predicting {i}')
        predict.run_yolov5_prediction('outputs/runs/pothole_yolov5s12/weights/best.pt', i, 'data/test/output')

run_prediction()


