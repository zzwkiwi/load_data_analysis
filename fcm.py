#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from skfuzzy.cluster import cmeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
import sqlite3

import data_read

def all_user_process(days):
    loads_df = data_read.data_process()
    # 按照日期和时间绘制数据透视表，获得不同时间下的用户用电数据
    loads_wide_df = pd.pivot_table(data=loads_df,columns=['date','day_of_month'],values='energy_use',index=['id'])

    loads_wide_df = loads_wide_df.T.iloc[0:-1]
    loads_wide_df = loads_wide_df.T
    unique_days = loads_df.day_of_month.unique()
    loads_wide_df = pd.concat([loads_wide_df.xs(days, level='day_of_month',axis=1) for day in unique_days])
    loads_wide_df=loads_wide_df.drop_duplicates()
    #查看缺失值，其中T代表将原有矩阵转置
    loads_wide_df.T.isnull().sum().sort_values(ascending=False).head()
    loads_wide_df = loads_wide_df.dropna(axis=0,how='any')
    return loads_wide_df

class EnergyFingerPrints():

    def __init__(self, data):
        # will contain the centroid of each cluster
        self.means = []
        self.data = data

    def elbow_method(self, n_clusters):

        fig, ax = plt.subplots(figsize=(8, 4))
        distortions = []

        for i in range(1, n_clusters):
            center, u, u0, d, jm, p, fpc = cmeans(self.data.T,
                                                  m=2,
                                                  c=i,
                                                  error=0.5,
                                                  maxiter=10000)

            label = np.argmax(u, axis=0)
            print('Cluster%s' % label)
            distortions.append(d)  # inertia计算样本点到最近的中心点的距离之和

        plt.plot(range(1, n_clusters), distortions, marker='o', lw=1)
        plt.xlabel('聚类数量')
        plt.ylabel('至中心点距离之和')
        plt.show()

    def get_cluster_counts(self):
        return pd.Series(self.predictions).value_counts()

    def labels(self, n_clusters):
        self.n_clusters = n_clusters
        center, u, u0, d, jm, p, fpc = cmeans(self.data.T, m=2, c=n_clusters, error=0.5, maxiter=10000)
        labels = np.argmax(u, axis=0)
        return labels

    def fit(self, n_clusters):
        """Performs K-means clustering for the load-profiles

        Parameters
        ----------
        n_clusters : int

        Returns
        --------
        count_dict : dict
            The number of load-profiles in each cluster
        """
        self.n_clusters = n_clusters
        center, u, u0, d, jm, p, fpc = cmeans(self.data.T, m=2, c=n_clusters, error=0.5, maxiter=10000)
        label = np.argmax(u, axis=0)
        self.predictions = label
        return self.predictions

def fcm_train(loads_wide_df):

    energy_clusters = EnergyFingerPrints(loads_wide_df)
    predictions = energy_clusters.fit(n_clusters=4)

    return predictions