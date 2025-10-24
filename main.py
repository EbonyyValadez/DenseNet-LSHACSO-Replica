import sys

import hpelm
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import optim, nn
# import  visdom
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
from pokemon0 import Pokemon
# from    resnet import ResNet18
from torchvision.models import resnet18
import os
from utils import Flatten

try:
    from mealpy.swarm_based.MPA import MPA
except ImportError:
    from mealpy.swarm_based.MPA import OriginalMPA as MPA

from mealpy.utils.problem import FloatVar


import pickle
from torchvision import transforms, models
from    PIL import Image
def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    total_val =0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        total_val += y.size(0)
        correct += torch.eq(pred, y).sum().float().item()
# y.size(0)
    return correct / total

def get_evaluate_acc_pred(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    total_val = 0
    predictions = []  # 存储所有的预测结果

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())  # 将预测值转换为 numpy 数组并添加到列表中

        total_val += y.size(0)
        correct += torch.eq(pred, y).sum().float().item()

    accuracy = correct / total
    return accuracy, predictions
def getevaluteY(model, loader):
    pre_Y = []
    Y = []
    model.eval()

    correct = 0
    total = len(loader.dataset)
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            # 将预测的Y和实际的Y追加到列表中
            pre_Y.extend(pred.cpu().numpy())
            Y.extend(y.cpu().numpy())
        correct += torch.eq(pred, y).sum().float().item()

    return pre_Y, Y

import random
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained = True)
        
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
            
        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()
        
        
    def get_optimizer(self):
        return torch.optim.AdamW([
            {'params' : self.base.parameters(), 'lr': 3e-5},
            {'params' : self.block.parameters(), 'lr': 8e-4}
        ])
        
        
    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x
class Densenet169(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.densenet169(pretrained=True)
        
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
            
        self.block = nn.Sequential(
            nn.Linear(1664, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()
        
        
    def get_optimizer(self):
        return torch.optim.AdamW([
            {'params' : self.base.parameters(), 'lr': 3e-5},
            {'params' : self.block.parameters(), 'lr': 8e-4}
        ])
        
        
    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x

def set_seed(seed):
    random.seed(seed)                       # 设置Python的随机种子
    np.random.seed(seed)                    # 设置NumPy的随机种子
    torch.manual_seed(seed)                 # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)            # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)        # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False
from torch.utils.data import DataLoader, SubsetRandomSampler
import random


# 训练新的模型
def generativeModel():
    global device, x_train, y_train

    # Set the seed for reproducibility
    
    batchsz = 16
    lr = 1e-3
    epochs =30
    num_cuda_devices = torch.cuda.device_count()
    print(f"当前系统上有 {num_cuda_devices} 个可用的CUDA设备。")

    # 指定要使用的CUDA设备
    desired_device_id = 0  # 选择要使用的设备的ID
    if desired_device_id < num_cuda_devices:
        torch.cuda.set_device(desired_device_id)
        print(f"已将CUDA设备切换到设备ID为 {desired_device_id} 的设备。")
    else:
        print(f"指定的设备ID {desired_device_id} 超出可用的CUDA设备数量。")
    device = torch.device('mps')
    parent_dir = os.path.dirname(os.getcwd())
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取当前脚本文件的父文件夹
    cwd_dir = os.path.dirname(script_path)

    

    
    model_name = ["densenet169_model"]
    for index  in range(0,1):
        
        val_acc_Trial = np.zeros((5, epochs))
        train_acc_Trial = np.zeros((5, epochs))
        val_loss_Trial = np.zeros((5, epochs))
        train_loss_Trial = np.zeros((5, epochs))
        test_acc_list=np.zeros((5, 1))
        for ii in range(0, 5):  
            set_seed(42+index+ii )
            if model_name[index]=="densenet169_model":
                model=Densenet169().to(device)

            else:
                pass



            
            print(f"执行模型{model_name[index]}:第{ii}次 -------------")
            
            filemame = f"images{ii}.csv"
            train_transform = transforms.Compose([
                # lambda x:Image.open(x).convert('RGB'),
                transforms.ToTensor(),
                transforms.Resize(size = (224, 224)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomVerticalFlip(p = 0.5)
            ])

            val_transform = transforms.Compose([
                    # lambda x:Image.open(x).convert('RGB'),
                transforms.ToTensor(),
                transforms.Resize(size = (224, 224))
            ])
            train_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='train',tran=train_transform)
            val_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='val',tran=val_transform)
            test_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='test',tran=val_transform)
                       

            # Create indices and random sampler
            indices = np.arange(len(train_db))
            indices1 = np.arange(len(val_db))
            indices2 = np.arange(len(test_db))
            # Use SubsetRandomSampler with a random seed
            sampler = SubsetRandomSampler(indices )
            sampler1 = SubsetRandomSampler(indices1)
            sampler2 = SubsetRandomSampler(indices2)

            # 假设 train_db, val_db, test_db 是你已经定义好的数据集
            # batchsz = 32

            
            # 在DataLoader中设置generator参数
            train_loader = DataLoader(train_db, batch_size=batchsz, sampler=sampler)
            val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=False)
            test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=False)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criteon = nn.CrossEntropyLoss()

            best_acc, best_epoch = 0, 0
            global_step = 0
            
            for epoch in range(epochs):
                correct_train = 0  # 用于计算训练集上的准确率
                total_train = 0  # 训练样本总数
                train_loss = 0  # 用于计算训练集损失

                for step, (x, y) in enumerate(train_loader):
                    # x: [b, 3, 224, 224], y: [b]
                    x, y = x.to(device), y.to(device)

                    model.train()
                    logits = model(x)

                    loss = criteon(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 累加训练损失
                    train_loss += loss.item()
                    _, preds = torch.max(logits, 1)  # 获取每个样本的预测标签
                    correct_train += (preds == y).sum().item()
                    total_train += y.size(0)  # 累加样本数量

                    global_step += 1

                # 计算当前 epoch 的训练集平均损失和准确率
                train_acc = correct_train / total_train
                avg_train_loss = train_loss / len(train_loader)

                # 验证阶段
                model.eval()
                val_loss = 0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(device), val_y.to(device)

                        logits = model(val_x)

                        loss = criteon(logits, val_y)

                        # 累加验证集损失
                        val_loss += loss.item()

                        # 计算验证集准确率
                        _, val_preds = torch.max(logits, 1)
                        correct_val += (val_preds == val_y).sum().item()
                        total_val += val_y.size(0)

                # 计算验证集的平均损失和准确率
                avg_val_loss = val_loss / len(val_loader)
                val_acc = correct_val / total_val

                # 存储准确率和损失
                val_acc_Trial[ii, epoch] = val_acc
                train_acc_Trial[ii, epoch] = train_acc
                val_loss_Trial[ii, epoch] = avg_val_loss
                train_loss_Trial[ii, epoch] = avg_train_loss

                if epoch % 1 == 0:

                    # val_acc = evalute(model, val_loader)
                    if val_acc > best_acc:
                        best_epoch = epoch
                        best_acc = val_acc
                        dirp = cwd_dir
                        if os.path.exists(os.path.join(dirp,model_name[index],str(epochs),"50dim")) == False:
                            os.makedirs(os.path.join(dirp,model_name[index],str(epochs),"50dim"))
                        torch.save(model.state_dict(), f'{dirp}/{model_name[index]}/{str(epochs)}/50dim/best{ii}.mdl')
                         

                    # viz.line(np.array([val_acc]), np.array([global_step]), win='val_acc', update='append')
                print("epoch:", {epoch}, ":best_acc", {best_acc})
                # 打印当前 epoch 的结果
                print(f"Epoch [{epoch}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}") 
            

                
                
               


from collections import defaultdict


def default_dict_factory():
    return defaultdict(dict)


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import cauchy
from copy import deepcopy
  
DimSize = 100
PopSizeMax = 20
PopSizeMin = 4
PopSize = PopSizeMax
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 1
MaxFEs = 100

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1
HistorySize = 5
mu_phi = np.array([0.3] * HistorySize)
elms=[]


def meanL(arr):
    numer = 0
    denom = 0
    for var in arr:
        numer += var ** 2
        denom += var
    return numer / denom



# initialize the M randomly
def Initialization(func):
    global Pop, Velocity, FitPop, PopSize, PopSizeMax, mu_phi, HistorySize, curFEs,elms
    PopSize = PopSizeMax
    Pop = np.zeros((PopSize, DimSize))
    FitPop = np.zeros(PopSize)
    Velocity = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] ,telm= func(Pop[i])
        elms.append(telm)
        curFEs += 1
    mu_phi = np.array([0.3] * HistorySize)


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def LSHACSO(func):
    global Pop, Velocity, FitPop, PopSize, curFEs, mu_phi,elm, x_train, y_train, x_val, y_val,y_test,elms

    sequence = list(range(PopSize))
    np.random.shuffle(sequence)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    Xmean = np.mean(Pop, axis=0)
    Success_phi = []
    r1 = np.random.randint(HistorySize)
    for i in range(int(PopSize / 2)):
        idx1 = sequence[2 * i]
        idx2 = sequence[2 * i + 1]

        if FitPop[idx1] < FitPop[idx2]:
            Off[idx1] = deepcopy(Pop[idx1])
            FitOff[idx1] = FitPop[idx1]

            phi = np.clip(np.random.normal(mu_phi[r1], 0.1, DimSize), 0.001, 0.5)
            Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (Pop[idx1] - Pop[idx2]) + phi * (Xmean - Pop[idx2])
            Off[idx2] = Pop[idx2] + Velocity[idx2]
            Off[idx2] = Check(Off[idx2])
            FitOff[idx2] ,t= func(Off[idx2])
            elms[idx2]=t
            curFEs += 1
            if FitOff[idx2] < FitPop[idx2]:
                Success_phi.append(np.mean(phi))
            
        else:
            Off[idx2] = deepcopy(Pop[idx2])
            FitOff[idx2] = FitPop[idx2]

            phi = np.clip(np.random.normal(mu_phi[r1], 0.1, DimSize), 0.001, 0.5)
            Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (Pop[idx2] - Pop[idx1]) + phi * (Xmean - Pop[idx1])
            Off[idx1] = Pop[idx1] + Velocity[idx1]
            Off[idx1] = Check(Off[idx1])
            FitOff[idx1] ,t= func(Off[idx1])
            elms[idx1]=t
            curFEs += 1
            if FitOff[idx1] < FitPop[idx1]:
                Success_phi.append(np.mean(phi))

    PopSize = round(((PopSizeMin - PopSizeMax) / MaxFEs * curFEs + PopSizeMax))
    if PopSize % 2 == 1:
        PopSize += 1

    c = 0.1
    if len(Success_phi) > 0:
        mu_phi[r1] = (1-c) * mu_phi[r1] + c * meanL(Success_phi)

    PopSize = max(PopSize, PopSizeMin)
    sorted_idx = np.argsort(FitOff)
    Pop = deepcopy(Off[sorted_idx[0:PopSize]])
    FitPop = deepcopy(FitOff[sorted_idx[0:PopSize]])
    t=[]
    for i in sorted_idx[0:PopSize]:
        t.append(elms[i])
    elms=t
device = torch.device('mps')
def main():
    import numpy as np
    global x_train, y_train, x_val, y_val, device,elms
    lr = 1e-3
    batchsz = 64

    parent_dir = os.path.dirname(os.getcwd())
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取当前脚本文件的父文件夹
    cwd_dir = os.path.dirname(script_path)
    
    result = defaultdict(default_dict_factory)


    model_name = [ "densenet169_model"]
    epochs=30 #加载运行了epochs的模型
    for index  in range(len(model_name)): 
        
        All_Trial_Best = []
        elm_acc = []

        for ii in range(0, 5):

            set_seed(42+index+ii )
             
             
            if model_name[index]=="densenet169_model":
                model=Densenet169().to(device)
            else:
                pass

            print(f"执行模型{model_name[index]}:第{ii}次 -------------")
            
            filemame = f"images{ii}.csv"

            train_transform = transforms.Compose([
            # lambda x:Image.open(x).convert('RGB'),
            transforms.ToTensor(),
            transforms.Resize(size = (224, 224)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5)
        ])

            val_transform = transforms.Compose([
                # lambda x:Image.open(x).convert('RGB'),
            transforms.ToTensor(),
            transforms.Resize(size = (224, 224))
        ])
            train_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='train',tran=train_transform)
            val_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='val',tran=val_transform)
            test_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='test',tran=val_transform)
        

            # Create indices and random sampler
            indices = np.arange(len(train_db))
            indices1 = np.arange(len(val_db))
            indices2 = np.arange(len(test_db))
            # Use SubsetRandomSampler with a random seed
            sampler = SubsetRandomSampler(indices)

            # 假设 train_db, val_db, test_db 是你已经定义好的数据集


            # 在DataLoader中设置generator参数
            train_loader = DataLoader(train_db, batch_size=batchsz,  sampler=sampler)
            val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=False)
            test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=False)
             
            # 加载已经训练好的模型的值
            model.load_state_dict(torch.load(f'{cwd_dir}/{model_name[index]}/{str(epochs)}/50dim/best{ii}.mdl'))

            # 存储生成的特征的路径，下次get_features就直接读取现成的特征文件
            file_path = os.path.join(cwd_dir, f'data/x_train{ii}_{model_name[index]}_50dim.pkl')
            file_path1 = os.path.join(cwd_dir, f'data/y_train{ii}_{model_name[index]}_50dim.pkl')
            file_path2 = os.path.join(cwd_dir, f'data/x_val{ii}_{model_name[index]}_50dim.pkl')
            file_path3 = os.path.join(cwd_dir, f'data/y_val{ii}_{model_name[index]}_50dim.pkl')
            file_path4 = os.path.join(cwd_dir, f'data/test{ii}_{model_name[index]}_50dim.pkl')
            file_path5 = os.path.join(cwd_dir, f'data/test_y{ii}_{model_name[index]}_50dim.pkl')


            # 获取训练，验证数据和测试数据
            train, train_y = get_features(model, train_loader, file_path, file_path1,model_name[index])
            val, val_y = get_features(model, val_loader, file_path2, file_path3,model_name[index])
            test, test_y = get_features(model, test_loader, file_path4, file_path5,model_name[index])

            # 不合并训练集和验证集
            x_train = train
            y_train = train_y
            x_val, y_val = val, val_y
            test, test_y = test, test_y
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.preprocessing import MinMaxScaler
            # Initialize the scaler with the desired range
            scaler = MinMaxScaler(feature_range=(0, 1 ))  # Default is (0, 1)

            # Fit and transform the training data
            x_train = scaler.fit_transform(x_train)

            # For the test data (use the same scaler fitted on training data)
            x_val = scaler.transform(x_val)
            test = scaler.transform(test)
            # 假设 x_train 和 x_val 是原始数据
            # 1. 标准化数据：PCA对数据的尺度敏感，通常我们先标准化数据。
            scaler = StandardScaler()

            # 标准化训练集和验证集
            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.transform(x_val)
            test_scaled = scaler.transform(test)
            # 如果提取的特征和标签不存在则保存，省的下次重新提取
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as file:
                    pickle.dump(x_train, file)
            if not os.path.exists(file_path1):
                with open(file_path1, 'wb') as file:
                    pickle.dump(y_train, file)
            if not os.path.exists(file_path2):
                with open(file_path2, 'wb') as file:
                    pickle.dump(x_val, file)
            if not os.path.exists(file_path3):
                with open(file_path3, 'wb') as file:
                    pickle.dump(y_val, file)
            if not os.path.exists(file_path4):
                with open(file_path4, 'wb') as file:
                    pickle.dump(test, file)
            if not os.path.exists(file_path5):
                with open(file_path5, 'wb') as file:
                    pickle.dump(test_y, file)



            from keras.utils import to_categorical
            import numpy as np
            from sklearn.metrics import accuracy_score
            # 生成onehot标签供elm训练
            y_train = to_categorical(y_train, 4)
            y_test = to_categorical(test_y, 4)
            y_val = to_categorical(val_y, 4)

            # d：设置神经元个数，与维度一起使用，例如d*512,代表有d*512个神经元
            #
            # n(输入维度): 这里的n = 512， 代表输入数据的维度（即特征数）。
            # m(输出维度): m = 10 代表神经网络的输出维度，m表示类别的数量。
            global d, n, m,elm
            d =2
            n = 128  # Input dimension
            m = 4  # Output dimension


            # Calculate total dimensions: weights + biases
            dimensions = int(d * n) * n + int(d * n)
            lb = np.ones(dimensions)*0
            ub = np.ones(dimensions)*1

            global curFEs, curFEs, TrialRuns, Pop, FitPop, DimSize,LB,UB,elms
            DimSize = dimensions
            LB =lb
            UB =ub
            import sys

            # 确保输出到标准输出而不是关闭的文件
            sys.stdout = sys.__stdout__
            for i in range(TrialRuns):
                Best_list = []
                curFEs = 0
                # np.random.seed(1996 + 30 * i)
                Initialization(objective_functionELM)
                Best_list.append(min(FitPop))
                while curFEs < MaxFEs:
                    LSHACSO(objective_functionELM)
                    Best_list.append(min(FitPop))
                    print("bestfit=", min(FitPop))
                All_Trial_Best.append(Best_list)


            val_y_pred=elms[0].predict(x_val)
            val_acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_y_pred, axis=1))
            # 预测结果
            y_pred = elms[0].predict(test)  # 假设X是输入数据
            # 计算准确率
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            
            elm_acc.append(acc)
            result[f"LSHACSO_{model_name[index]}_ELM"]["test"][f"preY{ii}"] = np.argmax(y_pred, axis=1)
            result[f"LSHACSO_{model_name[index]}_ELM"]["test"][f"Y{ii}"] = test_y
            result[f"LSHACSO_{model_name[index]}_ELM"]["test"]["accuracy"] = acc

            result[f"LSHACSO_{model_name[index]}_ELM"]["val"][f"preY{ii}"] = np.argmax(val_y_pred,axis=1)
            result[f"LSHACSO_{model_name[index]}_ELM"]["val"][f"Y{ii}"] = val_y
            result[f"LSHACSO_{model_name[index]}_ELM"]["val"]["accuracy"] = val_acc



            print(f"LSHACSO_{model_name[index]}_ELM test=:", acc)
            print(f"LSHACSO_{model_name[index]}_ELM val=:", val_acc)



            val_acc,val_pred_y0 = get_evaluate_acc_pred(model, val_loader)
            test_acc ,test_pred_y0= get_evaluate_acc_pred(model, test_loader)


            print(f"{model_name[index]}:test_acc={test_acc}")                 
            print(f"{model_name[index]}:val_acc={val_acc}")

            result[f"{model_name[index]}"]["val"][f"preY{ii}"] = val_pred_y0
            result[f"{model_name[index]}"]["val"][f"Y{ii}"] = val_y
            result[f"{model_name[index]}"]["val"]["accuracy"] = val_acc

            result[f"{model_name[index]}"]["test"][f"preY{ii}"] = test_pred_y0
            result[f"{model_name[index]}"]["test"][f"Y{ii}"] = test_y
            result[f"{model_name[index]}"]["test"]["accuracy"] = test_acc


            with open(f"{cwd_dir}/LSHACSO_output_50dim_re.txt", "a") as f:
                sys.stdout = f  # 临时重定向标准输出
                print(f"{model_name[index]}:test_acc={test_acc}")                 
                print(f"{model_name[index]}:val_acc={val_acc}")
                print(f"LSHACSO_{model_name[index]}_ELM test=:",acc)
            sys.stdout = sys.__stdout__
        np.savetxt(f"{cwd_dir}/LSHACSO/LSHACSO_{model_name[index]}" + "_" + str(DimSize) + "dim.csv", All_Trial_Best,
               delimiter=",")
        print(f"{model_name[index]}:平均精度：", np.mean(elm_acc))
    result_dict = convert_defaultdict_to_dict(result)
    # 存储数据到result
    with open(f"{cwd_dir}/LSHACSO_acc000.pkl", 'wb') as file:
        pickle.dump(result_dict, file)



def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    else:
        return d


def objective_functionELM(particle):
    global d, n, m, TZ
    global x_train, y_train, x_val, y_val, concatenate_x_train, concatenate_y_train, tag

    # 把粒子转换成权重和阈值矩阵
    # W = particle[:d * n * n].reshape(d * n, n)  # 假设有d*n个sigm神经元
    # B = particle[d * n * n:d * n * n + d * n].reshape(d * n, 1)  # 假设有d*n个sigm神经元

    # W = particle[:n * num_neurons].reshape(n, num_neurons)  # Weights: shape (512, 100)
    # B = particle[n * num_neurons:n * num_neurons + num_neurons].reshape(1, num_neurons)

    # W = X[:n * num_neurons].reshape(n, num_neurons)  # Weights: shape (512, 100)
    # B = X[n * num_neurons:n * num_neurons + num_neurons].reshape(num_neurons, 1)
    # num_neurons=100
    # W = X[:num_neurons * n].reshape(n, num_neurons)  # Extract and reshape the weight matrix W (shape: d x n)
    # B = X[n * num_neurons:n * num_neurons + num_neurons].reshape(num_neurons, 1)
    # beta = particle[d * n * n + d * n:].reshape(m, d * n)  # 假设有m个输出神经元
    # 创建一个ELM对象，指定输入和输出的维度
    W = particle[:int(d * n) * n].reshape(int(d * n), n)  # 假设有d*n个sigm神经元
    B = particle[int(d * n) * n:int(d * n) * n + int(d * n)].reshape(int(d * n), 1)  # 假设有d*n个sigm神经元
    elm0 = hpelm.ELM(n, m)
    # 添加一些神经元，指定激活函数
    elm0.add_neurons(int(d * n), "sigm")
    # 设置ELM的权重和阈值
    elm0.W = W
    elm0.B = B

        
    from keras.utils import to_categorical
    # Asegurar que las etiquetas estén en formato one-hot
    if y_train.ndim == 1:
        y_train_onehot = to_categorical(y_train, num_classes=m)
    else:
        y_train_onehot = y_train

    # Entrenar el ELM antes de predecir
    try:
        elm0.train(x_train, y_train_onehot)
    except Exception as e:
        print("Advertencia: el entrenamiento del ELM falló con", str(e))

    # 预测结果
    y_pred = elm0.predict(x_val)  # 假设X是输入数据

    acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
    y = np.argmax(y_val, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    f1 = f1_score(y, y_pred, average='weighted')
    re = recall_score(y, y_pred, average='weighted')
    acc = acc
    # 返回误差值
    return -1 * (1 * acc + 0 * f1 + 0 * re), elm0




# get_features如果路径文件已经存在则读取后直接返回，否则提取model的特征和lable
def get_features0(model, train_loader, x_path, y_path):
    global device
    if (not os.path.exists(x_path)) or (not os.path.exists(y_path)):

        model0 = nn.Sequential(*list(model.children())[:-1],  # [b, 512, 1, 1]
                               Flatten(),  # [b, 512, 1, 1] => [b, 512]
                               ).to(device)
        model0.eval()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = model0(x)
                if step == 0:
                    result = features
                    result_y = y;
                else:
                    result = torch.cat([result, features], dim=0)
                    result_y = torch.cat([result_y, y], dim=0)
        result, result_y = result.cpu(), result_y.cpu()
        with open(x_path, 'wb') as file:
            pickle.dump(result, file)
        with open(y_path, 'wb') as file:
            pickle.dump(result_y, file)

        return result.numpy(), result_y.numpy()
    else:
        with open(x_path, 'rb') as file:
            result = pickle.load(file)
        with open(y_path, 'rb') as file:
            result_y = pickle.load(file)

        return result.numpy(), result_y.numpy()
# get_features如果路径文件已经存在则读取后直接返回，否则提取model的特征和lable
def get_features(model, train_loader, x_path, y_path,modelname):
    global device
    if (not os.path.exists(x_path)) or (not os.path.exists(y_path)):
        
        if modelname == 'inception_model':
            
            model.base.fc[4]=nn.Identity()
            model0 =model
        else:

           model.block[3]=nn.Identity()
           model0 = model
        model0.eval()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                
            # 计算训练集准确率
                if modelname == 'inception_model':
                    logits = model0(x)
                    features = logits  # 获取每个样本的预测标签
                else:
                    logits = model0(x)
                    features =logits  # 获取每个样本的预测标签
                
                if step == 0:
                    result = features
                    result_y = y;
                else:
                    result = torch.cat([result, features], dim=0)
                    result_y = torch.cat([result_y, y], dim=0)
        result, result_y = result.cpu(), result_y.cpu()
        with open(x_path, 'wb') as file:
            pickle.dump(result, file)
        with open(y_path, 'wb') as file:
            pickle.dump(result_y, file)

        return result.numpy(), result_y.numpy()
    else:
        with open(x_path, 'rb') as file:
            result = pickle.load(file)
        with open(y_path, 'rb') as file:
            result_y = pickle.load(file)

        return result.numpy(), result_y.numpy()


# get_features1 提取model的特征和lable，并返回result.numpy(),result_y.numpy()
def get_features1(model, train_loader):
    global device
    model0 = nn.Sequential(*list(model.children())[:-1],  # [b, 512, 1, 1]
                           Flatten(),  # [b, 512, 1, 1] => [b, 512]
                           ).to(device)
    model0.eval()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            features = model0(x)
            if step == 0:
                result = features
                result_y = y;
            else:
                result = torch.cat([result, features], dim=0)
                result_y = torch.cat([result_y, y], dim=0)
    result, result_y = result.cpu(), result_y.cpu()
    return result.numpy(), result_y.numpy()


 

if __name__ == '__main__':
    # generativeModel()
    main()
