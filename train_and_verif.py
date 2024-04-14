import numpy as np
import torch
import cv2
import torchvision.transforms
from numpy import mat, ones
from torch import nn
from torch.utils.data import DataLoader
import aff_resnet
from tqdm import tqdm


# 训练
def cross_train_verifi(train_loader, num_classes, fuse_type, verifi_Loader, save_path, iter_num):
    net_model = aff_resnet.resnet34(num_classes, fuse_type=fuse_type, small_input=False).train()
    if torch.cuda.is_available():  # GPU是否可用
        net_model = net_model.cuda()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(net_model.parameters(), lr=5e-3, momentum=0.9)
    accuracy = 0.0  # 保存精度
    # 训练
    for epoch in range(600):
        i = 0
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1), ncols=100) as tq:
            for imgae, label in tqdm(train_loader):
                if torch.cuda.is_available():
                    imgae = imgae.cuda()
                    label = label.cuda()
                out = net_model(imgae)
                pred = out[0]
                loss = criterion(pred, label.long())
                i += 1
                tq.set_postfix({'lr': '%.5f' % optimizer.param_groups[0]['lr'], 'loss': '%.4f' % (loss.item())})
                tq.update(1)
                torch.cuda.empty_cache()
                # 反向传播，更新参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 交叉熵验证
        if epoch + 1 >= 200 and (epoch + 1) % 50 == 0:
            print("开始验证......")
            eval_f1 = verifi_start(net_model, verifi_Loader, save_path, iter_num)
            net_model.train()
            if str(eval_f1[2]) == 'nan':
                print('验证精度结果异常，继续训练')
            elif eval_f1[2] >= accuracy:
                print(eval_f1[0], eval_f1[1], eval_f1[2])
                accuracy = eval_f1[2]
                torch.save(net_model.state_dict(), '%s\\model\\down_best_model_%s.pth' % (save_path, iter_num))
    if accuracy == 0:
        print('模型训练失败，程序退出')
        exit(0)
    else:
        print('训练完毕，返回模型')
        net_model.load_state_dict(torch.load('%s\\model\\down_best_model_%s.pth' % (save_path, iter_num)))

    eval_f1 = verifi_start(net_model, verifi_Loader, save_path, iter_num)
    print(eval_f1[0], eval_f1[1], eval_f1[2])
    return net_model


# 验证
def verifi_start(net_model, verifi_Loader, save_path, iter_num):
    net_model.eval()  # 验证模式
    true_label_set = 0
    predict_label_set = 0
    id = 0
    for image, label in verifi_Loader:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        torch.cuda.empty_cache()
        output = net_model(image)
        predict_label = output[0].detach().max(1)[1]  # 预测验证集数据的概率
        if id == 0:
            true_label_set = label.reshape(1, -1)
            predict_label_set = predict_label.reshape(1, -1)
            id = 1
        else:
            label = label.reshape(1, -1)
            predict_label = predict_label.reshape(1, -1)
            true_label_set = torch.cat((true_label_set, label), dim=-1)
            predict_label_set = torch.cat((predict_label_set, predict_label), dim=-1)

    true_label_set = true_label_set.t().cpu()
    predict_label_set = predict_label_set.t().cpu()

    return eval.evaluation(true_label_set, predict_label_set, save_path, iter_num)
