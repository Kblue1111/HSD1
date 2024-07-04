import os
import copy
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from tool import Tool_csv
from tool import Tool_io
from model.LSTM import LSTM
# from multiprocessing import Pool
from model.BPSSModel import BPSSModel
from dataset import DatasetDeal
from dataset.DatasetModule import MyDataset
from dataset.DatasetDeal import deal_dataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.multiprocessing import Pool

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


# 遍历程序，训练，预测，统计结果
def process_start(pro_dataset, args):

    pro_names = Tool_io.get_folder(args.data_path)
    # 遍历不同程序
    for pro_name in pro_names:
        if pro_name != 'Lang':
            continue
        print(pro_name)
        # 程序所有版本
        ver_list = pro_dataset[pro_name]
        pro_path = os.path.join(args.data_path, pro_name)
        deal_dataset(ver_list, args.features, args.formulas, pro_path, args.coverage_file, args.origin, pro_name)
        # 划分训练集(验证集)、测试集
        data_list = DatasetDeal.leave_one(ver_list)

        # try:
        torch.multiprocessing.set_start_method('spawn')
        pool = Pool(processes=8)
        for number in range(len(data_list)):
            param = []
            param.append(data_list)
            param.append(number)
            param.append(pro_name)
            param.append(pro_path)
            param.append(args)
            # multi_process(param)
            # print("s")
            pool.apply_async(multi_process, (param,))
        pool.close()
        pool.join()
        # print("current program end")
        # except Exception as e:
        #     print(e)

        Tool_csv.cal_avg(args.res_path, pro_name)


def multi_process(param):
    # 获取参数
    data_list = param[0]
    number = param[1]
    pro_name = param[2]
    pro_path = param[3]
    args = param[4]
    print(pro_name,number)
    # 训练
    train_flag = 0
    model_path = os.path.join(args.model_path, pro_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_res = os.path.join(model_path, pro_name + '_' + str(number) + '.pkl')
    if os.path.exists(model_res):
        train_flag = 1

    if train_flag == 0:
        train_origin = data_list[number]['train']
        train_dataset = MyDataset(train_origin, pro_name, args.origin, data_list[number]['test'], args.coverage_file,
                                  pro_path, True)
        train_iter = DatasetDeal.get_dataloader(train_dataset, args.batch_size, True)
        train_process(train_iter, None, args, model_path, pro_name + '_' + str(number) + '.pkl', None)

    # 测试
    test_flag = 0
    vector_path = os.path.join(args.model_path, pro_name)
    vector_res = Tool_io.checkAndLoad(vector_path, pro_name + '_' + str(number) + '.td')
    if vector_res is not None:
        test_flag = 1

    if test_flag == 0:
        vector_res = {}
        for ver in data_list[number]['test']:
            test_dataset = MyDataset([ver], pro_name, args.origin, None, None, None, False)
            test_iter = DatasetDeal.get_dataloader(test_dataset, args.batch_size, False)
            res_lab, res_pre = predicate_process(test_iter, vector_path, pro_name + '_' + str(number) + '.pkl')
            res_lab = res_lab.cpu()
            res_pre = res_pre.cpu()
            vector_res[ver] = res_pre

            recall, precision, FPR, f1_score = cal_metric(res_lab, res_pre, pro_name)
            content = []
            content.append([ver, recall, precision, FPR, f1_score])
            Tool_csv.creat_res_file(args.res_path, pro_name, content)
        Tool_io.checkAndSave(vector_path, pro_name + '_' + str(number) + '.td', vector_res)


# 计算指标
def cal_metric(res_lab, res_pre, pro_name):
    res = confusion_matrix(res_lab, res_pre)
    cmatrix = classification_report(res_lab, res_pre)
    # 混淆矩阵参数
    TP = res[1, 1]
    TN = res[0, 0]
    FP = res[0, 1]
    FN = res[1, 0]
    # print(res)
    print(cmatrix)
    # cc识别指标
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if FP + TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    print(recall, precision, FPR, f1_score)
    return recall, precision, FPR, f1_score


# 训练模型
def train_process(train_iter, val_iter, args, model_path, pkl_name, change_val):
    # cnn
    # conv1_val = int(((41 - 3 + 1) - (3 - 1) - 1) / 3 + 1)
    # conv2_val = int(((conv1_val - 3 + 1) - (3 - 1) - 1) / 3 + 1)
    # lay_size = conv2_val * 64
    # model = CNNModel(1, 32, 64, 3, lay_size, 128).to(device)

    # mlp
    #model = BPSSModel(40, 1, 16, 128, 4, 32, 128, 2).to(device)

    # lstm
    model = LSTM(57, 128, 5, 2).to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_val)

    min_epochs = 10
    min_val_loss = 5
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = []
        for step, (sample, label) in enumerate(train_iter):
            sample = sample.float()
            sample = sample.to(device)
            label = label.to(device)
            y_pred = model(sample)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 10 == 0:
                print("epoch :{},step :{} ,Train loss: {}".format(epoch, step, np.sum(train_loss) / step))

        scheduler.step()
        # 验证
        # val_loss = get_val_loss(model, val_iter)
        # if epoch > min_epochs and val_loss < min_val_loss:
        #     min_val_loss = val_loss
        #     best_model = copy.deepcopy(model)
        # print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

    best_model = copy.deepcopy(model)
    # print('---------------------------------best_model:{}, min_val_loss:{}'.format(best_model, min_val_loss))
    print('---------------------------------best_model:{}'.format(best_model))
    # if (min_val_loss < change_val):
    #     state = {'models': best_model.state_dict()}
    #     torch.save(state, os.path.join(model_path, pkl_name))
    state = {'models': best_model.state_dict()}
    torch.save(state, os.path.join(model_path, pkl_name))
    return min_val_loss



# 测试过程
def predicate_process(test_iter, vector_path, td_name):

    # cnn
    # conv1_val = int(((41 - 3 + 1) - (3 - 1) - 1) / 3 + 1)
    # conv2_val = int(((conv1_val - 3 + 1) - (3 - 1) - 1) / 3 + 1)
    # lay_size = conv2_val * 64
    # model = CNNModel(1, 32, 64, 3, lay_size, 128).to(device)

    # mlp
    #model = BPSSModel(40, 1, 16, 128, 4, 32, 128, 2).to(device)

    # lstm
    model = LSTM(57, 128, 5, 2).to(device)

    path = os.path.join(vector_path, td_name)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()

    res_pre = torch.LongTensor(0).to(device)
    res_lab = torch.LongTensor(0).to(device)
    for (sample, target) in tqdm(test_iter):
        with torch.no_grad():
            sample = sample.float()
            sample = sample.to(device)
            target = target.to(device)
            out = model(sample)
            cal = nn.Softmax(dim=1)
            y_pred = cal(out)
            pre = torch.argmax(y_pred, dim=1)
            res_pre = torch.cat((res_pre, pre), 0)
            res_lab = torch.cat((res_lab, target), 0)

    return res_lab, res_pre



def get_val_loss(model, Val):
    val_loss = []
    model.eval()
    loss_function = nn.CrossEntropyLoss().to(device)
    print('validating...')
    for (sample, target) in Val:
        sample = sample.to(device)
        target = target.to(device)
        with torch.no_grad():
            sample = sample.float()
            y_pred = model(sample)
            loss = loss_function(y_pred, target)
            val_loss.append(loss.item())
    return np.mean(val_loss)


# def load_model(args):
#     model = BPModel(args.num_node, args.act_fun).to(device)
#     return model



class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss