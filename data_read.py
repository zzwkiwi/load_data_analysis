#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import sqlite3
import datademo
from datetime import datetime

def data_read():
    cwd = os.getcwd()
    conn = sqlite3.connect(str(cwd) + "/dataport_sqlite")
    cursor = conn.cursor()
    query = "SELECT * FROM new_table;"
    cursor.execute(query)
    data = cursor.fetchall()
    return data

def x_y_data_process(loads_df):
    df1 = loads_df['energy_use']
    sequence = 7
    X = []
    Y = []
    for i in range(df1.shape[0] - sequence):
        X.append(np.array(df1.iloc[i:(i + sequence)].values, dtype=np.float32))
        Y.append(np.array(df1.iloc[(i + sequence)], dtype=np.float32))
    X = np.array(X)
    Y = np.array(Y)
    # Y = Y.reshape(1,-1)
    return X,Y

def data_process():
    # data = data_read()
    # loads_df = pd.DataFrame(data, columns=['id', 'date', 'energy_use'])
    loads_df = datademo.data_read()
    # 查询数据中的空缺值
    loads_df = loads_df.replace('', np.nan)
    loads_df.isnull().sum()
    # 删除空缺值
    loads_df = loads_df.dropna()
    loads_df.time = loads_df.date.apply(lambda x: x.split()[1])
    localtime = datetime.now()
    loads_df = loads_df[loads_df['date'] < str(localtime)]

    # time = loads_df.time.drop_duplicates().sort_values()
    loads_df.loc[:, 'energy_use'] = loads_df.energy_use.astype(float)
    loads_df.loc[:, 'id'] = loads_df['id'].astype(int)
    loads_df.loc[:, 'date'] = pd.to_datetime(loads_df.date)
    # # 添加一代表星期的列
    # loads_df.loc[:, 'type_day'] = loads_df.date.apply(lambda x: x.isoweekday())
    # # 添加一代表日期的列
    loads_df.loc[:, 'day_of_month'] = loads_df.date.apply(lambda x: x.day)
    loads_df = loads_df.sort_values(['id', 'date'], ascending=[True, True])
    loads_df = loads_df.reset_index(drop=True)
    return loads_df


