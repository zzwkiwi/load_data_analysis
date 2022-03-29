#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import requests
import pandas as pd
def data_read():
    # req = requests.get('http://192.168.31.59:8021/energy/getAllUseEnergy')  # 请求连接
    req = requests.get('http://118.31.61.3:8020/energy/getAllUseEnergy')  # 请求连接

    req_jason = req.json()  # 获取数据
    df = pd.DataFrame([req_jason])
    df_result = df['result']
    loads_df = df_result[0]

    loads_df = pd.DataFrame(loads_df)
    loads_df.columns = ['date', 'energy_use', 'id']
    return loads_df


