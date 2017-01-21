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

df = pd.read_csv("../data/data-ss1.txt")
T = df.values[:,0].astype(np.float32)
Y = df.values[:,1].astype(np.float32)

n_times = len(df["X"].unique())

basic_model = Model()


#GaussianRandomWalkを使わない方法
with basic_model:
    #事前分布
    s_mu = HalfNormal('s_mu', sd=100) #隣接時刻の状態の誤差
    s_Y =  HalfNormal('s_Y', sd=100) #各時刻における状態と観測の誤差
    mu_0 = Normal('mu_0',mu=0, sd=100) #t=0初期状態
    mu_1 = Normal('mu_1',mu=0, sd=100) #t=1初期状態
    
    #誤差項
    e_mu = Normal('e_mu', mu=0, sd=s_mu, shape =n_times-2)
    
    mu = tt.zeros((n_times))
    mu = tt.set_subtensor(mu[0], mu_0)
    mu = tt.set_subtensor(mu[1], mu_1)
    for i in list(range(n_times-2)):
        mu = tt.set_subtensor(mu[i+2], 2*mu[i+1] - mu[i] + e_mu[i])
    #likelihood
    Y_obs = Normal('Y_obs', mu=mu, sd=s_Y, observed=Y)

    #サンプリング
    trace = sample(1000)
    summary(trace)