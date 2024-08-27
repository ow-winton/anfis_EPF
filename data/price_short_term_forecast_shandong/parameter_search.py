from pyswarm import pso
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

'''
from sklearn.preprocessing import LabelEncoder

# 创建 LabelEncoder 实例
label_encoder = LabelEncoder()



train_pure = pd.read_csv("../feature_data/train.csv", index_col=["timestamp"], parse_dates=["timestamp"])
test_pure = pd.read_csv("../feature_data/test.csv", index_col=["timestamp"], parse_dates=["timestamp"])

# 设置2023年的起始和结束日期
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31 23:59:59')

# 使用布尔索引筛选2023年的数据
train_data = train_pure[(train_pure.index >= start_date) & (train_pure.index <= end_date)]
test_data =test_pure


encoded_values = label_encoder.fit_transform(train_data['day_type'])
train_data['day_type_encoded'] = encoded_values
train_data = train_data.drop(columns=['day_type'])

encoded_values = label_encoder.fit_transform(test_data['day_type'])
test_data['day_type_encoded'] = encoded_values
test_data = test_data.drop(columns=['day_type'])

X_train_xgb = train_data
X = X_train_xgb.drop(columns=['实时电价'])
y = X_train_xgb['实时电价']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 使用示例
config = {
    'lb': [0.01, 3, 10],  # learning_rate, max_depth, n_estimators 的最小值
    'ub': [0.2, 10, 400],  # learning_rate, max_depth, n_estimators 的最大值
    'swarmsize': 30,
    'maxiter': 5,
    # 其他配置...
}

# 创建XGBModelOptimizer类的实例
optimizer = XGBModelOptimizer(config)

# 假设X_train, X_val, y_train, y_val已经定义
# optimizer.fit调用会进行模型的初始化、参数优化
optimizer.fit(X_train, X_val, y_train, y_val, config)

# 获取最佳参数
best_params = optimizer.get_best_params()
print(f'最佳参数: {best_params}')

# 使用最佳参数训练模型
optimizer.model.learning_rate, optimizer.model.max_depth, optimizer.model.n_estimators = best_params
optimizer.model.fit(X_train, y_train)

# 使用模型进行预测
predictions = optimizer.predict(X_val)


X_test_xgb  = test_data
X_test = X_test_xgb.drop(columns=['实时电价']) 

y_test_pred = optimizer.predict(X_test)
y_test_true = X_test_xgb['实时电价']

mae_test = mean_absolute_error(y_test_true, y_test_pred)
print(f'Test Mean Absolute Error: {mae_test}')

y_test_true_compute = y_test_true.reset_index(drop=True)
price_accuracy_spic_shandong(y_test_true_compute,y_test_pred)


model = XGBRegressor()
model.fit(X_train, y_train)
importances = model.feature_importances_

# 可视化特征重要性
plt.barh(range(len(importances)), importances, color='b')
plt.show()



'''


#
# class XGBModelOptimizer_v2:
#     def __init__(self, config):
#         self.config = config
#         self.model = None
#         self.best_params = None
#         self.best_score = None
#
#     def initialize_model(self):
#         # 使用配置初始化XGBRegressor模型
#         # 使用 **kwargs 来接收所有配置参数
#         kwargs = dict(
#             learning_rate=self.config.get('learning_rate', 0.1),
#             max_depth=self.config.get('max_depth', 6),
#             n_estimators=self.config.get('n_estimators', 100),
#             eval_metric='mae',
#             early_stopping_rounds=self.config.get('early_stopping_rounds', 8),
#             enable_categorical=self.config.get('enable_categorical', True),
#             categorical_features=self.config.get('categorical_features', ['day_type_encoded'])
#         )
#         self.model = XGBRegressor(**kwargs)
#
#     def objective_function(self, params):
#         # 优化的目标函数，接受任意数量的参数
#         # 假设params是一个包含learning_rate, max_depth, n_estimators的元组或列表
#         learning_rate, max_depth, n_estimators = params
#         self.model.learning_rate = learning_rate
#         self.model.max_depth = int(max_depth)
#         self.model.n_estimators = int(n_estimators)
#
#         self.model.fit(self.config['X_train'], self.config['y_train'],
#                        eval_set=[(self.config['X_val'], self.config['y_val'])],
#                        verbose=True)
#
#         y_val_pred = self.model.predict(self.config['X_val'])
#         mae = mean_absolute_error(self.config['y_val'], y_val_pred)
#
#         return mae
#
#     def optimize_parameters(self):
#         # 执行粒子群优化
#         lb, ub = self.config['lb'], self.config['ub']
#         # 假设lb和ub是长度相同的列表
#         self.best_params, self.best_score = pso(
#             self.objective_function,
#             lb,
#             ub,
#             swarmsize=self.config.get('swarmsize', 20),
#             maxiter=self.config.get('maxiter', 30)
#         )
#
#     def fit(self, X_train, X_val, y_train, y_val):
#         # 拟合模型的方法
#         self.config.update({
#             'X_train': X_train,
#             'X_val': X_val,
#             'y_train': y_train,
#             'y_val': y_val
#         })
#         self.initialize_model()
#         self.optimize_parameters()
#         # 转换best_params为浮点数列表
#         self.best_params = [float(i) for i in self.best_params]
#
#     def get_best_params(self):
#         # 获取最佳参数的方法
#         return self.best_params
#
#     def predict(self, X):
#         # 预测的方法
#         return self.model.predict(X)
#
#

class XGBModelOptimizer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.best_params = None
        self.best_score = None

    def initialize_model(self):
        # 使用配置初始化XGBRegressor模型
        self.model = XGBRegressor(
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=int(self.config.get('max_depth', 6)),
            n_estimators=self.config.get('n_estimators', 100),
            eval_metric='mae',
            early_stopping_rounds=self.config.get('early_stopping_rounds', 8),
            enable_categorical=self.config.get('enable_categorical', True),
            categorical_features=self.config.get('categorical_features', ['day_type_encoded'])
        )

    def objective_function(self, params):
        # 优化的目标函数
        learning_rate, max_depth, n_estimators = params
        self.model.learning_rate = learning_rate
        self.model.max_depth = int(float(max_depth))
        self.model.n_estimators = int(n_estimators)

        self.model.fit(self.config['X_train'], self.config['y_train'],
                       eval_set=[(self.config['X_val'], self.config['y_val'])],
                       verbose=True)

        y_val_pred = self.model.predict(self.config['X_val'])
        mae = mean_absolute_error(self.config['y_val'], y_val_pred)

        return mae

    def optimize_parameters(self):
        # 执行粒子群优化
        lb, ub = self.config['lb'], self.config['ub']
        self.best_params, self.best_score = pso(
            self.objective_function,
            lb,
            ub,
            swarmsize=self.config.get('swarmsize', 20),
            maxiter=self.config.get('maxiter', 2)
        )

    def fit(self, X_train, X_val, y_train, y_val, config):
        # 拟合模型的方法
        self.config.update({
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val
        })
        self.initialize_model()
        self.optimize_parameters()
        self.best_params = [float(self.best_params[0]), int(self.best_params[1]), int(self.best_params[2])]
        self.display_feature_importances()

    def get_best_params(self):
        # 获取最佳参数的方法
        return self.best_params

    def predict(self, X):
        # 预测的方法
        return self.model.predict(X)

    def display_feature_importances(self):
        # 检查模型是否已经训练
        if self.model and hasattr(self.model, 'feature_importances_'):
            # 获取特征重要性
            importances = self.model.feature_importances_
            # 获取特征名（假设您的DataFrame有列名）
            feature_names = self.config['X_train'].columns
            # 按照重要性排序
            sorted_indices = importances.argsort()
            # 打印特征重要性
            for i in sorted_indices[::-1]:  # 逆序排序
                print(f"{feature_names[i]}: {importances[i]}")
        else:
            print("Model not trained or feature_importances_ not available.")
