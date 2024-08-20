import sys
from Tool import DataProduction as dp
from train_and_verif import cross_train_verifi
from predict import *
from iteration_condition import whether_iteration

if __name__ == '__main__':
    # 参数设置
    orig_image = cv2.imread('读取数据影像')
    data_path = '读取数据路径' 
    sample_nums = '设置每一类的样本个数'
    class_num = '设置数据对应的类别个数'
    ##################################################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iter_num = 0  # 初始迭代次数
    wi_b = 1  # 初始化差异率
    while True:
        Train_iterNum = 'Train-%s' % iter_num
        # 加载训练数据
        is_bi_dataset = dp.ISBI_Loader(data_path, Train_iterNum, transform=torchvision.transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=is_bi_dataset, batch_size=16, shuffle=True)

        is_bi_dataset = dp.ISBI_Loader(data_path, 'ValidSet', transform=torchvision.transforms.ToTensor())
        val_loader = torch.utils.data.DataLoader(dataset=is_bi_dataset, batch_size=256, shuffle=False)

        # 模型
        print("开始训练.....")
        aff_resnet_model = cross_train_verifi(train_loader, class_num, val_loader, data_path, iter_num)

        is_bi_dataset = dp.ISBI_Loader(data_path, 'Test', transform=torchvision.transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=is_bi_dataset, batch_size=256, shuffle=False)
  
        guide_feature, pre_class_path, features_list, count_label = predict(aff_resnet_model, test_loader, class_num,
                                                                            iter_num, data_path)
        wi_a = whether_iteration(count_label, iter_num)
        print(abs(wi_a - wi_b))
        if abs(wi_a - wi_b) <= 0.01 or iter_num >= 10:
            sys.exit()
        else:
            wi_b = wi_a
        add_new_samples(guide_feature, pre_class_path, features_list, sample_nums, iter_num)
        iter_num += 1