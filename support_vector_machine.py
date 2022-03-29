#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,\
    median_absolute_error,explained_variance_score,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def svm_train(X, Y):
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, random_state=42)
    svr = SVR().fit(x_tr,y_tr)
    # 预测数据
    pre = svr.predict(X)
    return pre

def evaluate(y_te,pre):
    # print('线性回归模型的平均绝对误差', mean_absolute_error(y_te, pre))
    # print('线性回归模型的均方误差', mean_squared_error(y_te, pre))
    # print('线性回归模型的中值绝对误差', median_absolute_error(y_te, pre))
    # print('线性回归模型的可解释方差值', explained_variance_score(y_te, pre))
    r2 = r2_score(y_te, pre)
    y_var_te = y_te[1:] - y_te[:len(y_te) - 1]
    y_var_pre = pre[1:] - pre[:len(pre) - 1]
    txt = np.zeros(len(y_var_te))
    for i in range(len(y_var_te - 1)):
        txt[i] = np.sign(y_var_te[i]) == np.sign(y_var_pre[i])
    result = sum(txt) / len(txt)

    return float(r2), int(result * 100)