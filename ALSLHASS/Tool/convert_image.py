import cv2
import numpy as np


def to_image(path_gt, path_main):
    image = cv2.imread(path_gt)
    image_size = image.shape
    target_img = np.zeros([image_size[0], image_size[1], 3], dtype=np.uint8)
    path_txt = path_main + "\\Test-best.txt"
    with open(path_txt, 'r', encoding='utf-8') as infile:
        result_list = infile.readlines()
    for i in range(len(result_list)):
        str_1 = result_list[i].split(',')
        str_2 = str_1[len(str_1) - 1].split('-')
        str_3 = str_2[len(str_2) - 1].split('.')
        label = int(str_1[0])
        x = int(str_2[len(str_2) - 2])
        y = int(str_3[0])
        target_img[x, y, 0] = label  # 三通道赋值
        target_img[x, y, 1] = label  # 三 通道赋值
        target_img[x, y, 2] = label  # 三通道赋值
    cv2.imwrite(path_main + "\\Proposed.bmp", target_img)

