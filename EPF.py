'''
1. 划分验证集
2. 数据random shuffle
3. 保存检查点
4. 保存最好检查点
5. 尝试10个epoch 及不同特征
6. 保存log日志

'''

# 1. import package
import torch
from caffe2.python.layers import model_layer_subcls
from pyexpat import features
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import anfis
from membership import make_gauss_mfs, make_anfis
import experimental
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experimental import load_checkpoint






# 2.Load Data
data_path =r"C:\Users\bo.pan\PycharmProjects\anfis\data\morefeature\real_time.csv"
# feature = ['新能源', '系统负荷', '联络线', '日前电价', '负荷-联络线', '负荷-新能源', '新能源-联络线', '竞价空间']
'''
使用八个特征的同时用96个epoch 会爆内存，说需要912G内存，就用五个特征先跑吧。记得对比有没有normalize的区别
'''
feature = ['新能源', '系统负荷', '联络线', '日前电价','竞价空间']
# 读取2024以前的所有数据进来，使用特征是这几个单纯的供需特征加上日前电价的特征
oriData = pd.read_csv(data_path)
ln= len(oriData)
Data = oriData[:ln-20448]
train_dataset, dev_dataset = train_test_split(Data, test_size=0.2, random_state=42)
# print(len(train_dataset),val_dataset)


batchsize = 96
scaler = StandardScaler()


# Train data
train_Feature_Data =train_dataset[feature]
train_Label_Data = train_dataset['实时电价'].values.reshape(-1, 1)
train_numpy_array = train_Feature_Data.values
# 归一化
train_numpy_array_scaled = scaler.fit_transform(train_numpy_array)
train_label_array_scaled = scaler.fit_transform(train_Label_Data)
#转换为tensor
train_x_tensor = torch.from_numpy(train_numpy_array_scaled).float()
train_y_tensor = torch.tensor(train_label_array_scaled).float()
train_D = TensorDataset(train_x_tensor,train_y_tensor)
train_DataLoader = DataLoader(train_D,batch_size=batchsize,shuffle=True)
train_x,train_y = train_DataLoader.dataset.tensors
# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)

# Dev data
dev_Feature_Data =dev_dataset[feature]
dev_Label_Data = dev_dataset['实时电价'].values.reshape(-1, 1)
dev_numpy_array = dev_Feature_Data.values
# 归一化
dev_numpy_array_scaled = scaler.fit_transform(dev_numpy_array)
dev_label_array_scaled = scaler.fit_transform(dev_Label_Data)
#转换为tensor
dev_x_tensor = torch.from_numpy(dev_numpy_array_scaled).float()
dev_y_tensor = torch.tensor(dev_label_array_scaled).float()
dev_D = TensorDataset(dev_x_tensor,dev_y_tensor)
dev_DataLoader = DataLoader(dev_D,batch_size=batchsize,shuffle=True)
dev_x,dev_y = dev_DataLoader.dataset.tensors
# print(dev_x.shape,dev_y.shape)
# print(dev_x,dev_y)




# Model training
model = make_anfis(x=train_x,num_mfs=5,num_out=1)
optimizer = torch.optim.Rprop(model.parameters(),lr= 1e-4) # 优化器可以试试别的优化器
'''
Rprop（Resilient Backpropagation
SGD 也可以试试
'''
criterion = torch.nn.MSELoss() # 可以实时别的函数，具体的损失值可以考虑用sum来算


experimental.train_anfis_with_dev_data(model, train_DataLoader,dev_DataLoader, optimizer, criterion, 1)




# MODEL Testing

test_dataset = oriData[ln-20448:]
# Test data : 需要更换数据集
test_Feature_Data =test_dataset[feature]
test_Label_Data = test_dataset['实时电价'].values.reshape(-1, 1)
test_numpy_array = test_Feature_Data.values
# 归一化
test_numpy_array_scaled = scaler.fit_transform(test_numpy_array)
test_label_array_scaled = scaler.fit_transform(test_Label_Data)
#转换为tensor
test_x_tensor = torch.from_numpy(test_numpy_array_scaled).float()
test_y_tensor = torch.tensor(test_label_array_scaled).float()
test_D = TensorDataset(test_x_tensor,test_y_tensor)
test_DataLoader = DataLoader(test_D,batch_size=batchsize,shuffle=True)
test_x,test_y = test_DataLoader.dataset.tensors
# print(test_x.shape,test_y.shape)
# print(test_x,test_y)


model = make_anfis(x=train_x,num_mfs=5,num_out=1)
optimizer = torch.optim.Rprop(model.parameters(),lr= 1e-4) # 优化器可以试试别的优化器
from etide.model_evaluation import price_accuracy_spic_shandong
best_model, optimizer, start_epoch = load_checkpoint('best_checkpoint.pth.tar', model, optimizer)
y_test_pred  = best_model(test_x)
# 假设 y_actual 和 ypred 是需要梯度的 Tensor
# y_actual_detached = test_y.detach()


# 确保 y_test_pred 不需要梯度
y_test_pred_detached = y_test_pred.detach()
# 将 PyTorch 张量转换为 NumPy 数组
y_test_pred_np = y_test_pred_detached.numpy()
predictions_original = scaler.inverse_transform(y_test_pred_np)

# y_pred_np = pd.Series(predictions_original.flatten()).copy()

# 如果 predictions_original 是二维数组，需要将其扁平化为一维数组
if predictions_original.ndim == 2 and predictions_original.shape[1] == 1:
    predictions_original = predictions_original.flatten()

# 现在 predictions_original 已经是一维数组，可以直接用于评估
# 将预测结果转换为 Pandas Series，如果需要的话
y_pred_series = pd.Series(predictions_original.copy())

# label_original = scaler.inverse_transform()
print(price_accuracy_spic_shandong(test_dataset['实时电价'].values, y_pred_series))
## 归一化--> 实现了，但是逆归一化算回来的时候的结果并不完全跟原来的一样，可能差一点点，所以用的是test_dataset['实时电价']这种方法去写