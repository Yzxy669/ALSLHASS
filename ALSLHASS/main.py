from Tool import DataProduction as dp
from train_and_verif import cross_train_verifi
from predict import *

if __name__ == '__main__':
    # 参数设置
    orig_image = cv2.imread('Data path') # 读取影像
    data_path = 'Data saving path'  # 数据保存的主干路径
    sample_nums =  sample_nums  # 设置初始样本个数
    class_num = class_num  # 数据类别个数
    fuse_type = 'AFF'  # 注意力特征融合类型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iter_num = 0  
    # 加载训练数据
    Train_iterNum = 'Train-%s' % iter_num
    isbi_dataset = dp.ISBI_Loader(data_path, Train_iterNum, transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=32,
                                               shuffle=True)
    print("训练集个数：", len(isbi_dataset))

    isbi_dataset = dp.ISBI_Loader(data_path, 'ValidSet', transform=torchvision.transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                             batch_size=128,
                                             shuffle=False)
    print("验证集个数：", len(isbi_dataset))

    # 模型
    print("开始训练.....")
    aff_resnet_model = cross_train_verifi(train_loader, class_num, fuse_type, val_loader, data_path, iter_num)

    isbi_dataset = dp.ISBI_Loader(data_path, 'Test', transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                              batch_size=128,
                                              shuffle=False)
    print("测试集个数：", len(isbi_dataset))
    print("开始测试.....")
    guide_feature, pre_class_path, features_list = predict(aff_resnet_model, test_loader, class_num, iter_num,
                                                           data_path)
  
