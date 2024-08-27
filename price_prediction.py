import torch
from caffe2.python.layers import model_layer_subcls
from pyexpat import features
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import anfis
from membership import make_gauss_mfs, make_anfis
import experimental

def num_cat_correct(model, x, y_actual):
    '''
        Work out the number of correct categorisations the model gives.
        Assumes the model is producing (float) scores for each category.
        Use a max function on predicted/actual to get the category.
    '''
    y_pred = model(x)
    # Change the y-value scores back into 'best category':
    cat_act = torch.argmax(y_actual, dim=1)
    cat_pred = torch.argmax(y_pred, dim=1)
    num_correct = torch.sum(cat_act == cat_pred)
    return num_correct.item(), len(x)

# 数据要自己导入一下
data_path =r"C:\Users\bo.pan\PycharmProjects\anfis\data\morefeature\real_time.csv"
# Data = pd.read_csv(data_path,index_col=['timestamp'],parse_dates=['timestamp'])
oriData = pd.read_csv(data_path)
# print(Data.columns)
ln= len(oriData)
Data = oriData[:ln-20448]
# print(Data)

# feature = ['新能源','系统负荷','联络线','日前电价','负荷-联络线','负荷-新能源','新能源-联络线','竞价空间']
'''
考虑加上一些日前的特征？
'''
feature = ['新能源', '系统负荷', '联络线', '日前电价','竞价空间']
# feature = ['新能源', '系统负荷', '联络线', '日前电价','竞价空间']
feature_Data  = Data[feature]
label_Data = Data['实时电价'].values.reshape(-1,1)

# print(feature_Data)
# print(label_Data)
numpy_array = feature_Data.values
# print(numpy_array)
x_tensor = torch.from_numpy(numpy_array).float()
# print(x_tensor.shape)
y_tesor = torch.tensor(label_Data).float()
# print(y_tesor.shape)

td = TensorDataset(x_tensor,y_tesor)
train_Data = DataLoader(td,batch_size=20,shuffle=False)

x, y_actual = train_Data.dataset.tensors


# print(x)
# print(x.shape)
# print(y_actual)
# print(y_actual.shape)



model = make_anfis(x,num_mfs=5,num_out= 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
optimizer = torch.optim.Rprop(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss(reduction='sum')
experimental.train_anfis_with(model, train_Data, optimizer, criterion, 10)
# experimental.plot_all_mfs(model, x)



# print(model.coeff)




# 测试部分


# Data = pd.read_csv(data_path,index_col=['timestamp'],parse_dates=['timestamp'])
test_Data = oriData[ln-20448:]
# print(Data.columns)

# print(Data)
feature = ['新能源', '系统负荷', '联络线', '日前电价','竞价空间']
# feature = ['新能源','系统负荷','联络线','日前电价','负荷-联络线','负荷-新能源','新能源-联络线','竞价空间']
'''
考虑加上一些日前的特征？
'''
feature_Data  = test_Data[feature]

label_Data = test_Data['实时电价'].values.reshape(-1,1)
numpy_array = feature_Data.values
# print(numpy_array)
x_tensor = torch.from_numpy(numpy_array).float()
# print(x_tensor.shape)
y_tesor = torch.tensor(label_Data).float()

# print(y_tesor.shape)

td = TensorDataset(x_tensor,y_tesor)
train_Data = DataLoader(td,batch_size=20,shuffle=False)

x, y_actual = train_Data.dataset.tensors


# print(x)
# print(x.shape)
# print(y_actual)
# print(y_actual.shape)


nc, tot = num_cat_correct(model, x, y_actual)
# print('{} of {} correct (={:5.2f}%)'.format(nc, tot, nc*100/tot))
from etide.model_evaluation import price_accuracy_spic_shandong
ypred = model(x)
# 假设 y_actual 和 ypred 是需要梯度的 Tensor
y_actual_detached = y_actual.detach()
ypred_detached = ypred.detach()


# print(ypred)
# print(ypred_detached)
y_pred_np = ypred_detached.numpy()
y_pred_np = pd.Series(y_pred_np.flatten()).copy()
# print(y_pred_np)
# 现在调用函数，传入分离了梯度的张量


# print('shape')
# print(test_Data['实时电价'].values)
# print(y_pred_np)


print(price_accuracy_spic_shandong(test_Data['实时电价'].values, y_pred_np))


# 1. 归一化，
# 2. 96个合成一批
# 3.