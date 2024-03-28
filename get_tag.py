import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

def get_image_paths(folder_path):  
    image_paths = []  
    valid_images = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']  
    for ext in valid_images:  
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))  
    return image_paths

def write_tag(image_path,tag):
    # 获取图片文件名（不包括扩展名）  
    image_name = os.path.splitext(os.path.basename(image_path))[0]  
    # 构建文本文件的完整路径  
    text_file_path = os.path.join(folder_path, image_name + '.txt') 
    # 将内容写入文本文件  
    with open(text_file_path, 'w', encoding='utf-8') as text_file:  
        text_file.write(tag)  
        print(f"Text file created: {text_file_path}") 

def set_model(pretrained,vit='swin_l',image_size = 512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size)
    #######load model
    model = ram(pretrained = pretrained,
                             image_size = image_size,
                             vit = vit) # swin_l,swin_b
    model.eval()

    model = model.to(device)
    return transform,model

def get_image_tag(image_path,transform,model):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    res = inference(image, model)
    res_string = res[0].replace('|', ',') 
    return res_string

def get_all_tag(image_dir,pretrained,vit='swin_l',image_size = 512):
    transform,model = set_model(pretrained,vit,image_size)
    image_paths = get_image_paths(image_dir)
    for image_path in image_paths:
        tag = get_image_tag(image_path,transform,model)
        write_tag(image_path,tag) 
