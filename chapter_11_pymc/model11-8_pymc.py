# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3 import Model,Lognormal,Constant,Dirichlet,Categorical
from pymc3 import NUTS, sample, CategoricalGibbsMetropolis,find_MAP
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary
import theano.tensor as tt
import theano as T

df = pd.read_csv("../data/data-lda.txt")

n_person = len(df["PersonID"].unique())
n_item =120
K = 6
IDs = df.values[:,0].astype(np.int32)-1
Items = df.values[:,1].astype(np.int32)-1

#shape = クラスの数の確率変数に、クラスの値を取るデータ数次元のベクトルを入れる操作がありますが
#その詳細な説明は(https://pymc-devs.github.io/pymc3/notebooks/GLM-hierarchical.html)参照

basic_model = Model()
with basic_model:
    #事前分布[50,6]
    theta = Dirichlet('p_theta', a=(1.0/K)*np.ones(K),shape=(n_person,K))
    #事前分布[6,112]
    phi = Dirichlet('p_phi', a=(1.0/n_item)*np.ones(n_item), shape=(K,n_item))
    
    #likelihood
    #データ数 x 各データのカテゴリー確率ベクトル [1117,6]
    theta = theta[IDs,:] 
    #データ数 x 各IDに対するアイテム確率ベクトル [1117,112]
    person_to_item = tt.dot(theta, phi)
    
    H = Categorical("tes", p=person_to_item, shape=(1117), observed=Items)

    #サンプリング #パラメータの数が多く、ローカルで実行するには重いのでサンプリング数はかなり少なくしてます。
    #もしサンプル数を大きくしたければ、HのCategoricalを連続値にしてADVIをおすすめします。
    #連続値にするには、log-likelihoodを自分で定義すれば良いですが、その方法は下記に。
    #https://pymc-devs.github.io/pymc3/notebooks/lda-advi-aevb.html
    step = CategoricalGibbsMetropolis(vars=[H])
    trace = sample(12, init=None, step=step)
    summary(trace)