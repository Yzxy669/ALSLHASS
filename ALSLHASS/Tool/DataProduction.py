import cv2
import torch
from torch.utils.data import Dataset
import random
import glob
import os
import numpy as np
import torchvision


class ISBI_Loader(Dataset):
    # 初始化函数，读取所有DataPathName下的图片
    def __init__(self, data_path, data_type, transform):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.data_type = data_type
        self.images_path = glob.glob(os.path.join('%s\\%s\\*.png' % (data_path, data_type)))
        self.transform = transform

    def augment(self, TrainImage, flipCode):
        # 使用cv2.flip进行数据增强，fillipCode为1水平翻转，0为垂直翻转，-1水平翻转
        flip = cv2.flip(TrainImage, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图像
        if self.data_type == 'Test':
            image_path = self.images_path[index]
            image = cv2.imread(image_path, flags=-1)
            images = self.transform(image)
            return images, image_path
        else:
            image_path = self.images_path[index]
            image = cv2.imread(image_path, flags=-1)
            size = image.shape
            img = image[:, :, 0:size[2] - 1]
            label = image[:, :, size[2] - 1]
            img = self.transform(img)
            # 随机进行数据增强，为2时不处理
            # flipCote = random.choice([-1, 0, 1, 2])
            # if flipCote != 2:
            #   image = self.augment(image, flipCote)
            return img, label[0][0] - 1

    # 返回训练集大小
    def __len__(self):
        return len(self.images_path)
