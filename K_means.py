#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
import sqlite3
from sklearn.cluster import KMeans
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
        """Performs elbow method for a predefined number
        of clusters.

        Parameters
        ----------
        n_clusters : int
            the number of clusters to perform the elbow method

        Returns
        ---------
        A plot the of elbow method
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        distortions = []

        for i in range(1, n_clusters):
            km = KMeans(n_clusters=i,
                        init='k-means++',  # 初始中心簇的获取方式，k-means++一种比较快的收敛的方法
                        n_init=10,  # 初始中心簇的迭代次数
                        max_iter=300,  # 数据分类的迭代次数
                        random_state=0)  # 初始化中心簇的方式
            km.fit(self.data)
            distortions.append(km.inertia_)  # inertia计算样本点到最近的中心点的距离之和

        plt.plot(range(1, n_clusters), distortions, marker='o', lw=1)
        plt.xlabel('聚类数量')
        plt.ylabel('至中心点距离之和')
        plt.show()

    def get_cluster_counts(self):
        return pd.Series(self.predictions).value_counts()

    def labels(self, n_clusters):
        self.n_clusters = n_clusters
        return KMeans(self.n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0).fit(self.data).labels_

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
        self.kmeans = KMeans(self.n_clusters)
        self.predictions = self.kmeans.fit_predict(self.data)
        return self.predictions

def k_means_train(loads_wide_df):

    energy_clusters = EnergyFingerPrints(loads_wide_df)
    predictions = energy_clusters.fit(n_clusters=4)

    return predictions





