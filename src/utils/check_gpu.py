import torch

def checkgpu():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    return torch.cuda.is_available()