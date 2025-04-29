import yaml
from src.data import data_loader
from src.utils import train_test_splitter, check_gpu
import os
from src.utils import generate_empty_dir
def train_and_predict (config_path:str = 'configs'):

    with open(config_path + '/data.yaml','r') as f:
        data_configs = yaml.safe_load(f)

    #Creating required folders for training
    generate_empty_dir.create_dir()

    #data_loader.process_data()
    
    Raw_images_list = data_loader.loadData('data/Raw/images')
    train_images, test_images = train_test_splitter.train_test_split(Raw_images_list, test_size = data_configs['split_ratio'])
    
    #generating augamented data
    data_loader.augament_and_save_data(train_images, 'train')
    data_loader.augament_and_save_data(test_images, 'val')

    if check_gpu.checkgpu():
        print('training started using GPU')
        os.system(f"python yolov5/train.py --img 640 --batch {data_configs['n_batch']} --epochs {data_configs['n_epochs']} --data configs/data.yaml --weights yolov5s.pt --name pothole_yolov5s --device 0") #Adjust the attributes accordingly here.
    else:
        os.system(f"python yolov5/train.py --img 640 --batch {data_configs['n_batch']} --epochs {data_configs['n_epochs']} --data configs/data.yaml --weights yolov5s.pt --name pothole_yolov5s")
        
    