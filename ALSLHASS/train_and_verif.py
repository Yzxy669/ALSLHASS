import torch
from torch import nn
from Tool import Evaluation as eval
from aff_resnet import *
from tqdm import tqdm


# 训练
def cross_train_verify(train_loader, num_classes, verify_Loader, save_path, iter_num):
    net_model = aff_net(num_classes, fuse_type='AFF', small_input=False).train()
    if torch.cuda.is_available():  # GPU是否可用
        net_model = net_model.cuda()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(net_model.parameters(), lr=5e-3, momentum=0.9)
    accuracy = 0.0  # 初始化精度
    # 训练
    for epoch in range(500):
        i = 0
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1), ncols=100) as tq:
            for image, label in tqdm(train_loader):
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()
                if image.size(0) == 1:
                    continue
                out = net_model(image)
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
        if epoch + 1 >= 50 and (epoch + 1) % 50 == 0:
            print("开始验证......")
            eval_f1 = verify_start(net_model, verify_Loader, save_path, iter_num)
            net_model.train()
            if eval_f1[0] > accuracy and str(eval_f1[2]) != 'nan':
                print(eval_f1[0], eval_f1[1], eval_f1[2])
                accuracy = eval_f1[0]
                torch.save(net_model.state_dict(), './model/down_best_model_%s.pth' % iter_num)
    print('训练完毕，返回模型')
    if accuracy == 0:
        exit('第%s次迭代模型训练失败，算法结束' % iter_num)
    net_model.load_state_dict(torch.load('./model/down_best_model_%s.pth' % iter_num))
    eval_f1 = verify_start(net_model, verify_Loader, save_path, iter_num)
    print(eval_f1[0], eval_f1[1], eval_f1[2])

    return net_model


# 验证
def verify_start(net_model, verify_Loader, save_path, iter_num):
    net_model.eval()  # 验证模式
    true_label_set = 0
    predict_label_set = 0
    id = 0
    for image, label in tqdm(verify_Loader):
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
