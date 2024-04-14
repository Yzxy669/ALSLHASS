import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

"""
true_label:真值
pre_label：预测标签
epoc：训练的次数
save_path：评价结果保存路径

"""


def evaluation(true_label, pre_label, save_path, iter_num):
    cm = confusion_matrix(true_label, pre_label)  # 得到混淆矩阵
    recall_class = []  # 存储每一类得召回率
    F1 = []  # 存储每一类的F1分数
    pre_class = []  # 存储的每一类精度
    total_correct = np.diag(cm)  # 对角线上的数据
    rol_vector = cm.sum(axis=1)  # 获取每一行元素的1*n矩阵
    col_vector = cm.sum(axis=0)  # 获取每一列元素的n*1矩阵
    pee = 0  # 临时变量
    po = sum(total_correct) / len(true_label)  # 计算kappa的变量，等于整体精度
    path = save_path + '\\Evaluation-%s.txt' % str(iter_num)  # 存储精度txt
    with open(path, 'w') as fw:
        fw.write('OA:整体精度\n')
        fw.write('Kappa:卡帕系数\n')
        fw.write('AA:平均用户精度\n')
        fw.write('F-1:F1分数\n\n')
        for i in range(0, cm.shape[0]):
            for j in range(0, cm.shape[1]):
                fw.write('%s' % str(cm[i][j]) + '\t')
            fw.write('%s' % str(rol_vector[i]) + '\t')
            pee += rol_vector[i] * col_vector[i]
            precision = round((total_correct[i] / rol_vector[i]) * 100, 2)
            pre_class.append(precision)
            fw.write('%s' % str(precision) + '\t')
            recall = round((total_correct[i] / col_vector[i]) * 100, 2)
            recall_class.append(recall)
            f1 = round(((2 * precision * recall) / (recall + precision)), 2)
            F1.append(f1)
            fw.write('%s' % str(f1) + '\t')
            fw.write('\n')

        for k in range(0, len(col_vector)):
            fw.write('%s' % str(col_vector[k]) + '\t')
        fw.write('\n')
        fw.write('\n')
        # 写每一类的召回率
        for cla in range(0, len(recall_class)):
            fw.write('%s' % str(recall_class[cla]) + '\t')
        fw.write('\n')
        OA = round(po * 100, 2)
        AA = round((sum(pre_class) / cm.shape[0]), 2)
        pe = pee / (len(true_label) * len(true_label))
        Ka = round((po - pe) / (1 - pe), 3)
        Macro_aver = round((sum(F1) / cm.shape[0]), 2)
        fw.write('OA = %s' % str(OA) + '\t')
        fw.write('Ka = %s' % str(Ka) + '\t')
        fw.write('AA = %s' % str(AA) + '\t')
        fw.write('Macro_aver = %s' % str(Macro_aver) + '\t')
    fw.close()
    return OA, AA, Macro_aver
