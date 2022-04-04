from re import X
from tensorflow.keras.models import load_model
import cv2, os
# print(cv2.__version__)
from http.client import HTTPResponse

import numpy as np
import matplotlib as plt

# # 모델이 keras CNN인 경우
def img_predict_keras(dog_breed, selected_model, decode_img, img_name):
    
    model_path = 'dog_models/'
    image_path = f'Image/{dog_breed}/saveimg'

    model = load_model(f'{model_path}/{selected_model}')

    cv2.imwrite(f'{image_path}/{img_name}', decode_img)

    img = decode_img / 255
    dst = cv2.resize(img, dsize=(700,700))
    test = (np.expand_dims(dst, 0))

    predict_prob = model.predict(test)
    if round(predict_prob[0][0],2) >= 0.5 :
        return "관리가 필요합니다"
    else :
        return "정상입니다"

## predict.py 안에서 TEST  (karas로 할 때 이미지 저장하는거 구현해야 함)
# import base64, io, cv2
# import numpy as np
# from PIL import Image
# def stringToRGB(base64_string):
#     imgdata = base64.b64decode(base64_string)
#     dataBytesIO = io.BytesIO(imgdata)
#     image = Image.open(dataBytesIO)
#     return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

# img_path = 'Image/Welsh Corgi/nor_308.jpg'
# img_name = 'fffff.jpg'

# with open(img_path, 'rb') as img:
#     base64_str = base64.b64encode(img.read())

# print( img_predict_keras('Welsh Corgi','corgi_model_4.h5',stringToRGB(base64_str), img_name) )



###### 모델이 Pytorch인 경우
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time

import os, shutil
import matplotlib as plt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

def trim(image):
    h, w = image.shape[0], image.shape[1]
    # The center of a image
    X, Y = int(w/2), int(h/2)

    if w > h :    # 폭 > 높이 : 가로 방향
        img_trim = image[ : ,  X-int(h/2) : X+int(h/2)  ]
    elif w < h :  # 폭 < 높이 : 세로 방향
        img_trim = image[  Y-int(w/2) : Y+int(w/2)  , : ]
    else :        # 폭 = 높이 : 정방형
        img_trim = image
    
    return img_trim 


def img_predict_torch(dog_breed, selected_model, decode_img, img_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체

    model_path = 'dog_models/'
    image_path = f'Image/{dog_breed}/saveimg'

    # 모델 업로드 
    model = torch.load(f'{model_path}/{selected_model}', map_location='cpu')
    
    if dog_breed == 'Retriever': 
        ## 이미지를 선명하게 만듦
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        image_sharp = cv2.filter2D(decode_img, -1, kernel)

    elif dog_breed == 'Dachshund':
        trim_img = trim(decode_img)
        image_yuv = cv2.cvtColor(trim_img, cv2.COLOR_BGR2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        image_sharp = cv2.filter2D(image_yuv, -1, kernel)

    cv2.imwrite(f'{image_path}/{img_name}', image_sharp)

    ## test 전처리 
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'Image'
    test_datasets = datasets.ImageFolder(os.path.join(data_dir, f'{dog_breed}'), transforms_test)

    class_names = ['비만', '정상']

    ## 이미지 출력 함수
    def imshow(input, title):
        # torch.Tensor를 numpy 객체로 변환
        input = input.numpy().transpose((1, 2, 0))
        # 이미지 정규화 해제하기
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input = std * input + mean
        input = np.clip(input, 0, 1)
        # 이미지 출력
        plt.imshow(input)
        plt.title(title)
        plt.show()

    ## 이미지 업로드 
    image = Image.open(f'{image_path}/{img_name}')
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return f'당신의 강아지는 {class_names[preds[0]]}입니다!'


# ## predict.py 안에서 TEST2  
# import base64, io, cv2
# import numpy as np
# from PIL import Image

# def stringToRGB(base64_string):
#     imgdata = base64.b64decode(base64_string)
#     dataBytesIO = io.BytesIO(imgdata)
#     image = Image.open(dataBytesIO)
#     return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

# img_path = 'images.jpeg'
# img_name = 'Pytorchtest.jpg'

# with open(img_path, 'rb') as img:
#     base64_str = base64.b64encode(img.read())

# print(img_predict_torch('Retriever','ret_set1_L6.pth',stringToRGB(base64_str), img_name) )


