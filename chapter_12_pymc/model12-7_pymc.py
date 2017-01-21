# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3 import Model, Normal, HalfNormal,Uniform
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary
import theano.tensor as tt
import theano as T
import math

df = pd.read_csv("../data/data-changepoint.txt")
T = df.values[:100,0].astype(np.float32)
Y = df.values[:100,1].astype(np.float32)
n_times = 100
#n_times = len(df["X"].unique()) #時間の数
basic_model = Model()

#subtensorの使い方↓
#http://deeplearning.net/software/theano/library/tensor/basic.html

with basic_model: 
    #事前分布
    #コーシー分布は逆関数法にて乱数生成
    s_mu =  HalfNormal('s_mu', sd=1) #コーシー分布の分散
    s_Y =  HalfNormal('s_Y', sd=1) #観測誤差
    mu_0 = Normal('mu_0',mu=0, sd=1)  #初期状態
    x = Uniform("x" ,lower=-math.pi/2,upper=math.pi/2, shape=n_times-1)
    
    #Cauchyの誤差process
    c = tt.dot(s_mu,tt.tan(x))
    
    #状態process
    mu = tt.zeros((n_times))
    mu = tt.set_subtensor(mu[0], mu_0)
    for i in range(n_times-1):
        mu = tt.set_subtensor(mu[i+1], mu[i]+c[i])

    #likelihood
    Y_obs = Normal('Y_obs', mu=mu, sd=s_Y, observed=Y)    
    
    #サンプリング 
    trace = sample(1000,n_init=5000)
    summary(trace)
