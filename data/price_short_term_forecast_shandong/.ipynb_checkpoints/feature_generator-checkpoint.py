from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from pyswarm import pso
import numpy as np
import time
import pandas as pd
import numpy as np
import pandas as pd
import math
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from pyswarm import pso
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from etide.feature_engineer import FeatureGenerator
from etide.model_evaluation import price_accuracy_spic_shandong
import os
import time
import datetime
from datetime import timedelta
import pandas as pd
from chinese_calendar import is_holiday,get_holiday_detail
from sklearn.preprocessing import LabelEncoder

def day_type(df2,df1):

    # 遍历 df2 并为每个匹配的日期设置 df1 中的 'day_type' 值
    for index, row in df2.iterrows():
        # 获取 df2 的日期部分
        date_str = index.date().isoformat()  # 将 date 对象转换为 ISO 格式的字符串

        # 在 df1 中查找匹配的 'day_type' 值
        # 注意我们使用字符串格式的日期，并且使用 .loc 来避免 SettingWithCopyWarning
        day_type_value = df1.loc[date_str, 'day_type']  # 假设列名是 'day_type_chinese'

        # 将找到的 'day_type' 值复制到 df2 的当前行
        df2.loc[index, 'day_type'] = day_type_value  # 假设我们要复制到 'day_type_chinese' 列

    return df2


def price_space_feature(df):
    # 为避免直接修改原始df，使用df.copy()来创建副本
    df_copy = df.copy()
    df_copy["负荷-联络线"] = df_copy["系统负荷"] - df_copy["联络线"]
    df_copy["负荷-新能源"] = df_copy["系统负荷"] - df_copy["新能源"]
    df_copy["新能源-联络线"] = df_copy["新能源"] - df_copy["联络线"]
    df_copy["竞价空间"] = df_copy["系统负荷"] - df_copy["新能源"] - df_copy["联络线"]
    return df_copy


def statistic_feature(df):
    train_pure = df.copy()
    # 假设FeatureGenerator类和generate_features方法已经被正确定义
    fg = FeatureGenerator()

    # 定义要生成的特征列表
    feature_functions = [
        'intraday_mean',
        'intraday_median',
        'intraday_max',
        'intraday_min',
    ]

    # 定义需要生成统计特征的列
    cols = [
        '新能源',
        '联络线',
        '实时电价',
        '负荷-联络线',
        '负荷-新能源',
        '新能源-联络线',
        '竞价空间',
    ]

    # 遍历每个列，使用generate_features方法生成特征
    for col in cols:
        train_pure = fg.generate_features(train_pure, feature_functions=feature_functions, cols=[col])

    # 返回修改后的DataFrame
    return train_pure


def timeset(df):
    data = df.copy()
    # 用来将时间戳数据转换，并且按照时间分组成清晨，白天，黑夜组， 增加这个特征来测试是否可以提高性能。
    data['hour'] = data.index.hour
    data['day'] = data.index.day
    data['weekday'] = data.index.weekday
    data['month'] = data.index.month
    data['is_morning'] = data['hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)
    data['is_daytime'] = data['hour'].apply(lambda x: 1 if 6 <= x < 18 else 0)
    data['is_evening'] = data['hour'].apply(lambda x: 1 if 18 <= x < 24 else 0)
    return data


def rolling_time_feature(df, time=8):
    data = df.copy()
    data['rolling_mean_8h_实时电价'] = data['实时电价'].rolling(window=time * 4).mean()  # 直接赋值，不使用.loc
    data['rolling_std_8h_实时电价'] = data['实时电价'].rolling(window=time * 4).std()
    data['rolling_min_8h_实时电价'] = data['实时电价'].rolling(window=time * 4).min()
    data['rolling_max_8h_实时电价'] = data['实时电价'].rolling(window=time * 4).max()

    data['rolling_mean_8h_新能源'] = data['新能源'].rolling(window=time * 4).mean()  # 直接赋值，不使用.loc
    data['rolling_std_8h_新能源'] = data['新能源'].rolling(window=time * 4).std()
    data['rolling_min_8h_新能源'] = data['新能源'].rolling(window=time * 4).min()
    data['rolling_max_8h_新能源'] = data['新能源'].rolling(window=time * 4).max()

    data['rolling_mean_8h_系统负荷'] = data['系统负荷'].rolling(window=time * 4).mean()  # 直接赋值，不使用.loc
    data['rolling_std_8h_系统负荷'] = data['系统负荷'].rolling(window=time * 4).std()
    data['rolling_min_8h_系统负荷'] = data['系统负荷'].rolling(window=time * 4).min()
    data['rolling_max_8h_系统负荷'] = data['系统负荷'].rolling(window=time * 4).max()

    data['rolling_mean_8h_联络线'] = data['联络线'].rolling(window=time * 4).mean()  # 直接赋值，不使用.loc
    data['rolling_std_8h_联络线'] = data['联络线'].rolling(window=time * 4).std()
    data['rolling_min_8h_联络线'] = data['联络线'].rolling(window=time * 4).min()
    data['rolling_max_8h_联络线'] = data['联络线'].rolling(window=time * 4).max()

    data['rolling_mean_8h_竞价空间'] = data['竞价空间'].rolling(window=time * 4).mean()  # 直接赋值，不使用.loc
    data['rolling_std_8h_竞价空间'] = data['竞价空间'].rolling(window=time * 4).std()
    data['rolling_min_8h_竞价空间'] = data['竞价空间'].rolling(window=time * 4).min()
    data['rolling_max_8h_竞价空间'] = data['竞价空间'].rolling(window=time * 4).max()
    return data


def diff_feature(data, periods=8):
    df= data.copy()
    # 计算实时电价的差分
    df['price_diff_新能源'] = df['新能源'].diff()
    # 计算多个时滞的差分
    df['price_diff_8h_新能源'] = df['新能源'].diff(periods=4 * periods)

    df['price_diff_系统负荷'] = df['系统负荷'].diff()
    # 计算多个时滞的差分
    df['price_diff_8h_系统负荷'] = df['系统负荷'].diff(periods=4 * periods)

    df['price_diff_联络线'] = df['联络线'].diff()
    # 计算多个时滞的差分
    df['price_diff_8h_联络线'] = df['联络线'].diff(periods=4 * periods)

    df['price_diff_竞价空间'] = df['竞价空间'].diff()
    # 计算多个时滞的差分
    df['price_diff_8h_竞价空间'] = df['竞价空间'].diff(periods=4 * periods)
    return df


def lag_feature(data, period=8):
    df= data.copy()
    df['lag_Feature_8h_新能源'] = df['新能源'].shift(period * 4)  # 直接赋值，不使用.loc
    df['lag_Feature_8h_系统负荷'] = df['系统负荷'].shift(period * 4)  # 直接赋值，不使用.loc

    df['lag_Feature_8h_联络线'] = df['联络线'].shift(period * 4)  # 直接赋值，不使用.loc
    df['lag_Feature_8h_竞价空间'] = df['竞价空间'].shift(period * 4)  # 直接赋值，不使用.loc
    return df

def division_feature(df):
    df_copy = df.copy()
    df_copy['竞价空间_divide_mean']= df_copy['竞价空间']/df_copy['竞价空间_intraday_mean']
    df_copy['竞价空间_divide_max'] = df_copy['竞价空间'] / df_copy['竞价空间_intraday_max']
    df_copy['竞价空间_divide_min'] = df_copy['竞价空间'] / df_copy['竞价空间_intraday_min']
    df_copy['竞价空间_divide_mid'] = df_copy['竞价空间'] / df_copy['竞价空间_intraday_median']
    return  df_copy



class feature_generation():
    def __init__(self):
        # self.train_path =  train_path
        # self.test_path = test_path
        # self.day_type_path = day_type_path
        return


    def train_data_load(self,path):
        data = pd.read_csv(path,index_col=["timestamp"], parse_dates=["timestamp"])
        return  data
    def date_data_load(self,path):
        data = pd.read_csv(path,index_col=["timestamp"], parse_dates=["timestamp"])
        return  data

    def date_feature_generator(self,df,df1):
        df = day_type(df, df1)
        return df
    def math_feature_generator(self,df):

        df = price_space_feature(df)
        df = statistic_feature(df)
        df = division_feature(df)
        df = timeset(df)
        df = rolling_time_feature(df)
        df = diff_feature(df)
        df = lag_feature(df)
        df.fillna(method='bfill', inplace=True)
        return df

    def generate(self,train_path,test_path,day_type_path):
        # 加载训练集，数据集
        train_data = self.train_data_load(train_path)
        test_data = self.train_data_load((test_path))
        date_type = self.date_data_load(day_type_path)


        train_data = self.date_feature_generator(train_data,date_type)
        test_data = self.date_feature_generator(test_data,date_type)

        train_data = self.math_feature_generator(train_data)
        test_data = self.math_feature_generator(test_data)
        return train_data,test_data






#     df1 = pd.read_csv("only_date_type.csv", index_col=["timestamp"], parse_dates=["timestamp"])
#     # 假设 df1 和 df2 都是已经存在的 DataFrame，并且它们的索引都是 datetime 类型。


