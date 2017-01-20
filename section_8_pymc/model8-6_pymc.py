# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary

df = pd.read_csv("data/data-salary-3.txt")
X_data = df.values[:,0]
Y_data = df.values[:,1]
company_data  = df.values[:,2]-1
cluster_data = df.values[:,3]-1
n_company = len(df["KID"].unique())
n_cluster = len(df["GID"].unique())

basic_model = Model()

#shape = クラスの数の確率変数に、クラスの値を取るデータ数次元のベクトルを入れる操作がありますが
#その詳細な説明は(https://pymc-devs.github.io/pymc3/notebooks/GLM-hierarchical.html)参照

#model8-5に業界毎の分散を追加
with basic_model:
    #全体平均
    a_0 = Normal('a_0', mu=0, sd=10)
    b_0 = Normal('b_0', mu=0, sd=10)
    #全体分散
    s_ag = HalfNormal('s_ag', sd=100)
    s_bg = HalfNormal('s_bg', sd=100)
    #業界平均
    a_g = Normal('a_g', mu=a_0, sd=s_ag, shape=n_cluster)
    b_g = Normal('b_g', mu=b_0, sd=s_bg, shape=n_cluster)
    #業界毎の誤差分散
    s_a = HalfNormal('sigma_a', sd=100,shape=n_cluster)
    s_a = s_a[cluster_data]
    s_b = HalfNormal('sigma_b', sd=100,shape=n_cluster)
    s_b = s_b[cluster_data]
    
    #個人の誤差分散
    s_Y = HalfNormal('sigma_Y', sd=100)
    
    #likelihood 
    #b = Normal('b', mu=b_g[cluster_data], sd=s_b)とするとエラーがでる
    a = Normal('a', mu=0, sd=s_a)+a_g[cluster_data]
    b = Normal('b', mu=0, sd=s_b)+b_g[cluster_data]
    mu = a+ b*X_data
    Y_obs = Normal('Y_obs', mu=mu, sd=s_Y, observed=Y_data)
    trace = sample(2000)
    summary(trace)