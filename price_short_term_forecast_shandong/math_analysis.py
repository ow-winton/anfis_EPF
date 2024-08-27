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

def feat_corr(input_df):
    corr = input_df.corr()
    plt.figure(figsize=(15, 12))
    # plot heat map
    g = sns.heatmap(corr, annot=True, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.title('Feature Correlation')

    return plt.show()
