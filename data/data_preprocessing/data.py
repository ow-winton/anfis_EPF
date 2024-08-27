import pandas as pd


data_path =r"../morefeature/real_time.csv"
# Data = pd.read_csv(data_path,index_col=['timestamp'],parse_dates=['timestamp'])
Data = pd.read_csv(data_path,index_col=['timestamp'],parse_dates=['timestamp'])
print(len(Data))
print(Data.columns)

label_path = r'C:\Users\bo.pan\PycharmProjects\anfis\data\labels.csv'

label = pd.read_csv(label_path,index_col=['timestamp'],parse_dates=['timestamp'])
print(len(label))


new = pd.merge(Data,label,how='inner',on=['timestamp'])
print(new.columns)
new.to_csv('data.csv')