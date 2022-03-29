#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as DTR #此处为回归决策树
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error,\
    median_absolute_error,explained_variance_score,r2_score

import data_read
import torch
def data_process():
    data = data_read.data_read()
    loads_df = pd.DataFrame(data, columns=['id', 'date', 'energy_use'])
    # 查询数据中的空缺值
    loads_df = loads_df.replace('', np.nan)
    loads_df.isnull().sum()
    # 删除空缺值
    loads_df = loads_df.dropna()
    loads_df.shape
    loads_df.time = loads_df.date.apply(lambda x: x.split()[1])
    time = loads_df.time.drop_duplicates().sort_values()
    loads_df.loc[:, 'energy_use'] = loads_df.energy_use.astype(float)
    loads_df.loc[:, 'id'] = loads_df['id'].astype(int)
    loads_df.loc[:, 'date'] = pd.to_datetime(loads_df.date)
    loads_df = loads_df.sort_values(['id', 'date'], ascending=[True, True])
    loads_df = loads_df.reset_index(drop=True)
    return loads_df

def tree_train(X, Y):
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, random_state=42)
    # 构建决策树模型 回归树
    dtc = DTR()
    # 训练模型 ,需要将type转换为int，否则报错
    dtc.fit(x_tr, y_tr.astype('int'))
    dtc.fit(x_tr, y_tr)
    # 预测数据
    # dtc = dtc = torch.load('decision_tree.pkl')
    pre = dtc.predict(X)
    return pre

def evaluate(y_te,pre):
    r2 = r2_score(y_te, pre)
    y_var_te = y_te[1:] - y_te[:len(y_te) - 1]
    y_var_pre = pre[1:] - pre[:len(pre) - 1]
    txt = np.zeros(len(y_var_te))
    for i in range(len(y_var_te - 1)):
        txt[i] = np.sign(y_var_te[i]) == np.sign(y_var_pre[i])
    result = sum(txt) / len(txt)

    return float(r2), int(result * 100)