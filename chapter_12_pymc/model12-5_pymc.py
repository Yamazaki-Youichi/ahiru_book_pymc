# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3 import Model, Normal, GaussianRandomWalk, HalfNormal
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary
import theano.tensor as tt
import theano as T

df = pd.read_csv("../data/data-ss2.txt")
T = df.values[:,0].astype(np.float32)
Y = df.values[:,1].astype(np.float32)

n_times = len(df["X"].unique()) #時間の数
L = 4 #トレンド周期

basic_model = Model()

#subtensorの使い方↓
#http://deeplearning.net/software/theano/library/tensor/basic.html

with basic_model:
    #事前分布
    s_mu = HalfNormal('s_mu', sd=100) #隣接時刻の状態の誤差
    s_season = HalfNormal('s_season', sd=100) #季節誤差
    s_Y =  HalfNormal('s_Y', sd=100) #各時刻における状態と観測の誤差
    season_0_1_2 = Normal('season_0_1_2',mu=0, sd=100, shape=3) #季節項の誤差なし初期値
    
    #季節process 
    ##まず初期状態の代入
    season = tt.zeros((n_times))
    for i in list(range(L-1)):
        season = tt.set_subtensor(season[i], season_0_1_2[i])
    
    ##初期状態以降の代入
    for t in list(range(n_times - 3)):
        # sum^{L-1}_{l=1} -season[t-l] のテンソルを作成
        Sigma = tt.zeros((1))
        for l in list(range(L-1)):
            Sigma +=  season[(t+3)-l]        
        ##時刻 t+3 に(誤差なし)状態を代入
        season =tt.set_subtensor(season[t+3], -Sigma[0])  
    ##最後に誤差を追加
    season = Normal("season", mu=season , sd=s_season, shape=n_times)

    #状態process
    mu = GaussianRandomWalk("mu",s_mu, shape=n_times )
        
    #likelihood
    Y_obs = Normal('Y_obs', mu=mu+season, sd=s_Y, observed=Y)

    #サンプリング
    trace = sample(100)
    summary(trace)
    
