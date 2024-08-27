# import torch
# print(torch.__version__)
import torch
from caffe2.python.layers import model_layer_subcls
from pyexpat import features

from torch.onnx.symbolic_opset9 import reshape
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import anfis
from membership import make_gauss_mfs, make_anfis
import experimental

# 数据要自己导入一下
data_path =r"C:\Users\bo.pan\PycharmProjects\anfis\data\morefeature\real_time.csv"
# Data = pd.read_csv(data_path,index_col=['timestamp'],parse_dates=['timestamp'])
oriData = pd.read_csv(data_path)
ln= len(oriData)
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
print(label_Data)
print(label_Data.shape)
a = label_Data.astype('float')

print(a.shape)