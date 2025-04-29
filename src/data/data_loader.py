import glob as glob
from src.data import data_augamentor
from src.utils import save_image_and_label as sil




def loadData(path: str) -> list:
    
    files = glob.glob(path + '/images/*.jpg')
    print(f'Total {len(files)} images have been found in raw data')
    return files
        

def get_image_label(path:str):

    label_file = path.replace('images', 'labels').replace('.jpg', '.txt')
    label_file = label_file.replace('png', 'txt')
    return  label_file


def augament_and_save_data(image_list: list, datapath:str) -> None:
    
    print(f'{len(image_list)} images have been alocated for {datapath}set')
    for i in range(len(image_list)):
        aug_img, aug_label, original_img, original_label = data_augamentor.augament_image_and_label(image_list[i])
        sil.save(str(i), aug_img, aug_label, f'/{datapath}/')
        sil.save('org_'+str(i), original_img, original_label, f'/{datapath}/')
    print(f"Succesfully generated {datapath} data")



