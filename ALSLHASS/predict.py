import math
import random
import torch
import glob
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import PIL.Image as Image

plt.switch_backend('agg')


#预测整张影像的标签
def predict(net_model, test_loader, class_num, iter_num, data_path):
    net_model.eval() 
    train_sample_path = data_path + '\\Train-%s' % (iter_num + 1)  # 初始化下一次训练样本的数据路径
    if not os.path.exists(train_sample_path):
        os.mkdir(train_sample_path)
    transform = torchvision.transforms.ToTensor()
    pre_class_path = [[] for i in range(class_num)] 
    features_list = [[] for i in range(class_num)]  
    guide_feature = [[] for i in range(class_num)]  
    count_label = []  
    ######################################################################################
    torch.cuda.empty_cache()
    guide_knn_path = glob.glob(os.path.join('%s\\Train-%s\\*.png' % (data_path, iter_num)))
    for path_i in zip(guide_knn_path):
        str = path_i[0].split('\\')
        str_1 = str[len(str) - 1].split('-')
        label = int(str_1[0]) - 1
        guide_knn_image = cv2.imread(path_i[0])
        image = transform(guide_knn_image)
        image = torch.unsqueeze(image, 0)
        output = net_model(image.cuda())
        _, feats = output[0], output[1]
        feats = feats.cuda().data.cpu().tolist()
        guide_feature[label].append(feats[0])
  
    torch.cuda.empty_cache()
    initial_sample_path = glob.glob(os.path.join('%s\\Train-0\\*.png' % data_path))
    for path_i in zip(initial_sample_path):
        # ===============将初始训练样本写入下一次迭代的训练样本文件中=============================================
        label_image = cv2.imread(path_i[0], flags=-1)
        label_image_path = path_i[0].replace('Train-0', 'Train-%s' % (iter_num + 1))
        cv2.imwrite(label_image_path, label_image)
        ######################################################################################
        str = path_i[0].split('\\')
        str_1 = str[len(str) - 1].split('-')
        label = int(str_1[0]) - 1
        initial_sample = cv2.imread(path_i[0])
        image = transform(initial_sample)  # 将Images影像的维度数轴变换
        image = torch.unsqueeze(image, 0)  # 将三维的Tensor包装成四维
        output = net_model(image.cuda())
        _, feats = output[0], output[1]
        feats = feats.cuda().data.cpu().tolist()  # 提取到的深度特征
        features_list[label].append(feats[0])
        pre_class_path[label].append(path_i[0])
    # =============================================测试数据分类=============================================
    for image, path_image in tqdm(test_loader):
        torch.cuda.empty_cache()
        output = net_model(image.cuda())
        pred, feats = output[0], output[1]
        feats = feats.cuda().data.cpu().tolist()  
        pred = pred.cuda().data.cpu().numpy()
        for num_image in range(len(path_image)):
            pre_label = int(np.argmax(pred[num_image], axis=0))
            count_label.append(pre_label)
            pre_class_path[pre_label].append(path_image[num_image])
            features_list[pre_label].append(feats[num_image])
    to_label_txt(pre_class_path, data_path, iter_num)

    return guide_feature, pre_class_path, features_list, count_label


# 选择新的训练样本加入训练集
def add_new_samples(guide_feature, pre_class_path, features_list, sample_nums, iter_num):
    print("开始增强样本.....")
    cl_image_path = [[] for i in range(len(features_list))]  # 存储每一类选择到的训练样本路径
    con_feature = [[] for i in range(len(features_list))]
    con_img_path = [[] for i in range(len(features_list))]
    # 构造KNN，决策树分类器的训练样本
    x_train_feature = []  
    y_train_label = []  
    for i in range(len(guide_feature)):
        for j in range(len(guide_feature[i])):
            x_train_feature.append(guide_feature[i][j])
            y_train_label.append(i)
    decision_tree = DecisionTreeClassifier()
    knn = KNeighborsClassifier(n_neighbors=3, p=2, weights='distance', algorithm='kd_tree')
    decision_tree.fit(x_train_feature, y_train_label)
    knn.fit(x_train_feature, y_train_label)
    # 选择置信度高的伪标签样本
    can_feature = [[] for i in range(len(features_list))]
    can_img_path = [[] for i in range(len(features_list))]

    for class_i in range(len(features_list)):
        c_image_path, c_feature = rough_sample(pre_class_path[class_i], features_list[class_i], sample_nums)
        # =====================为主动学习选择的样本提供标签==============================
        label_decision_tree = decision_tree.predict(c_feature)
        label_knn = knn.predict(c_feature)
        for i in range(len(label_knn)):
            if label_decision_tree[i] == class_i and label_knn[i] == class_i:
                can_feature[class_i].append(c_feature[i])
                can_img_path[class_i].append(c_image_path[i])

    for class_i in range(len(can_img_path)):
        sel_sample_num = int(len(can_img_path[class_i]) * 0.5)  # 根据设备算力设置每类地物随机选取的数量
        rand_num = random.sample(range(0, len(can_img_path[class_i])), sel_sample_num)
        for j in range(sel_sample_num):
            con_feature[class_i].append(can_feature[class_i][rand_num[j]])
            con_img_path[class_i].append(can_img_path[class_i][rand_num[j]])
    # 直方图和主动学习选择伪标签样本
    for class_i in range(len(features_list)):
        c_mean = np.mean(features_list[class_i][0:sample_nums], 0)  # 计算精确训练样本的特征均值
        m_c_feat = con_feature[class_i][:]
        m_c_feat.insert(0, c_mean)
        dist_vector = euclidean_distance(m_c_feat) 
        sample_feature, sample_image = similarity_histogram(dist_vector, con_feature[class_i], con_img_path[class_i])
        # 按照直方图及相似比例选择训练样本
        for bin_i in range(len(sample_feature)):
            # 将每个相似度bin下选择代表性的样本写入训练集
            top_res_feat, top_res_path = select_represent(sample_feature[bin_i], sample_image[bin_i])
            select_weights = 1 - bin_i / 10
            histogram_sample = len(sample_feature[bin_i]) * len(sample_feature[bin_i]) / len(con_feature[class_i])
            if bin_i == 0 and int(histogram_sample) <= 1:
                new_sample_num = int(len(sample_feature[bin_i]) * select_weights) - 1
            else:
                new_sample_num = int(histogram_sample * select_weights)
            for num in range(new_sample_num):
                cl_image_path[class_i].append(top_res_path[num])
    # =================================将训练样本写入文件============================================
    max_sample = 0
    for class_i in range(len(cl_image_path)):
        if len(cl_image_path[class_i]) > max_sample:
            max_sample = len(cl_image_path[class_i])
    for class_i in range(len(cl_image_path)):
        # 直接将每类训练样本写入文件
        for k in range(len(cl_image_path[class_i])):
            train_image_path = cl_image_path[class_i][k]
            image = cv2.imread(train_image_path)
            img_label = np.zeros([image.shape[0], image.shape[1], 1], dtype=np.uint8)
            img_label[np.where(img_label == 0)] = class_i + 1
            new_image = np.concatenate((image, img_label), axis=2)
            path = train_image_path.replace('Test', 'Train-%s' % (iter_num + 1))
            path_image = path.replace('unlabel', str(class_i + 1))
            cv2.imwrite(path_image, new_image)
        if 0 < len(cl_image_path[class_i]) < max_sample:
            need_enhance_num = max_sample - len(cl_image_path[class_i])
            while need_enhance_num > 0:
                angle = random.sample(range(0, 360), sample_nums)
                for j in range(sample_nums):
                    if need_enhance_num == 0:
                        break
                    train_image_path = pre_class_path[class_i][j]
                    image = cv2.imread(train_image_path)
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    rotation_image = image.rotate(angle[j])
                    r_rato, g_rato, b_rato = rotation_image.split()
                    merge_ratoimg = Image.merge("RGB", (b_rato, g_rato, r_rato))
                    img_label = np.zeros([image.height, image.width, 1], dtype=np.uint8)
                    img_label[np.where(img_label == 0)] = class_i + 1
                    new_image = np.concatenate((merge_ratoimg, img_label), axis=2)
                    path = train_image_path.replace('Train-0', 'Train-%s' % (iter_num + 1))
                    str_1 = path.split('.')
                    path_image = str_1[0] + '-rota-%s.png' % angle[j]
                    cv2.imwrite(path_image, new_image)
                    need_enhance_num -= 1


# ========================================================辅助函数================================================================
def euclidean_distance(vectors_list):
    vectors_list = np.array(vectors_list)
    vecA = vectors_list[0]
    ed_vector = []
    for i in range(1, len(vectors_list)):
        vecB = vectors_list[i]
        c = np.sqrt(np.sum(np.square(vecA - vecB)))
        ed_vector.append(c)
    return ed_vector


def Residual_distance(vectors_list):
    vectors_list = np.array(vectors_list)
    vecA = vectors_list[0]
    rd_vector = []
    for i in range(1, len(vectors_list)):
        vecB = vectors_list[i]
        c = 1 - np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        rd_vector.append(c)

    return rd_vector


def Manhattan_distance(vectors_list):
    vectors_list = np.array(vectors_list)
    vecA = vectors_list[0]
    Man_distance = []
    for i in range(1, len(vectors_list)):
        vecB = vectors_list[i]
        c = np.sum(np.abs(vecA - vecB))
        Man_distance.append(c)

    return Man_distance


def Chebyshev_distance(vectors_list):
    vectors_list = np.array(vectors_list)
    vecA = vectors_list[0]
    Che_distance = []
    for i in range(1, len(vectors_list)):
        vecB = vectors_list[i]
        c = np.max(np.abs(vecA - vecB))
        Che_distance.append(c)

    return Che_distance


def rough_sample(image_path, features, per_samples):
    feature_precise = np.mean(features[0:per_samples], 0)
    rf_feat = features[per_samples:len(features)]
    rf_feat.insert(0, feature_precise)
    dist_vector = euclidean_distance(rf_feat)
    coarse_sample_path, coarse_feature = remove_outliers(dist_vector, image_path,
                                                         features, per_samples)
    return coarse_sample_path, coarse_feature


def similarity_histogram(dist_vector, features, image_path):
    dis_norm = MaxMinNormalization(dist_vector)
    feature_bin = [[] for i in range(10)]
    image_path_bin = [[] for i in range(10)]
    for i in range(len(dis_norm)):
        bin = math.ceil(dis_norm[i] / 10) - 1
        if bin > 9:
            continue
        feature_bin[bin].append(features[i])
        image_path_bin[bin].append(image_path[i])
    return feature_bin, image_path_bin


def remove_outliers(dist_vector, image_path, features_list, per_samples):
    rs_image_path = []
    rs_feature = []
    temp_list = dist_vector[:]
    temp_list.sort()
    Q_1 = int(len(temp_list) * 0.6)
    for j in range(len(dist_vector)):
        if dist_vector[j] <= temp_list[Q_1]:
            rs_image_path.append(image_path[j + per_samples])
            rs_feature.append(features_list[j + per_samples])

    return rs_image_path, rs_feature


def feature_normalization(vector_list):
    features = np.matrix(vector_list[:])
    Max = features.max()
    Min = features.min()
    normal_feature = []
    for i in range(len(vector_list)):
        normalized_list = []
        for j in range(len((vector_list[i]))):
            x = (vector_list[i][j] - Min) / (Max - Min)
            normalized_list.append(x)
        normal_feature.append(normalized_list)
    return normal_feature


def MaxMinNormalization(vector_list):
    Max = max(vector_list)
    Min = min(vector_list)
    normalized_list = []
    for i in range(len(vector_list)):
        x = 100 * (vector_list[i] - Min) / (Max - Min)
        normalized_list.append(x)
    return normalized_list


def select_represent(sampel_feature_set, image_path_set):
    top_respersent_feature = []
    top_respersent_path = []
    repersentative_list = []
    for i in range(len(sampel_feature_set)):
        sampel_feature_list = sampel_feature_set[:]
        if i == 0:
            sample_i_ed = euclidean_distance(sampel_feature_list)
            represent_i = np.mean(sample_i_ed, 0)
            repersentative_list.append(represent_i)
        else:
            sample_i = sampel_feature_list[i]
            del (sampel_feature_list[i])
            sampel_feature_list.insert(0, sample_i)
            sample_i_ed = euclidean_distance(sampel_feature_list)
            represent_i = np.mean(sample_i_ed, 0)
            repersentative_list.append(represent_i)

    temp_repersentative_list = repersentative_list[:]
    temp_repersentative_list.sort()
    for i in range(len(temp_repersentative_list)):
        for j in range(len(repersentative_list)):
            if repersentative_list[j] == temp_repersentative_list[i]:
                top_respersent_feature.append(sampel_feature_set[j])
                top_respersent_path.append(image_path_set[j])
                break

    return top_respersent_feature, top_respersent_path


def to_label_txt(pre_class_path, save_path, iter_num):
    path = save_path + '\\Test-' + str(iter_num) + '.txt'
    with open(path, 'w') as f:
        for i in range(len(pre_class_path)):
            for j in range(len(pre_class_path[i])):
                f.write(str(i + 1) + ',')
                f.write(pre_class_path[i][j] + '\n')


def inspect(orig_image, check_image_path, class_num, data_path):
    color_list = [(0, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0),
                  (255, 0, 255), (0, 0, 255), (0, 0, 0), (0, 97, 255),
                  (203, 192, 255), (128, 148, 163), (42, 42, 128), (240, 32, 160),
                  (192, 192, 192), (15, 94, 56), (31, 102, 156), (240, 32, 160)]
    reslut = data_path + '\\reslut-%s.png' % class_num
    for i in range(len(check_image_path)):
        str = check_image_path[i].split('\\')
        str_1 = str[len(str) - 1].split('-')
        str_2 = str_1[len(str_1) - 1].split('.')
        x = int(str_1[1])
        y = int(str_2[0])
        around = [(x - 1, y), (x, y - 1), (x, y), (x, y + 1), (x + 1, y)]
        for j in range(5):
            if 0 <= around[j][0] < orig_image.shape[0] and 0 <= around[j][1] < orig_image.shape[1]:
                orig_image[around[j][0], around[j][1], 0] = color_list[class_num][0]
                orig_image[around[j][0], around[j][1], 1] = color_list[class_num][1]
                orig_image[around[j][0], around[j][1], 2] = color_list[class_num][2]
    cv2.imwrite(reslut, orig_image)
