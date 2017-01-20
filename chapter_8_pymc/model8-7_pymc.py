# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal,Lognormal
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary
import theano.tensor as tt

df = pd.read_csv("data/data-conc-2.txt")
df = df.drop("PersonID", axis=1)
Y_nt = df.values
n_patient = Y_nt.shape[0]
times = np.reshape(np.array([1,2,4,8,12,24]),(1,-1))

basic_model = Model()

with basic_model:
    #全体平均
    a_0 = Normal('a_0', mu=0, sd=10)
    b_0 = Normal('b_0', mu=0, sd=10)
    #全体分散
    s_ag = HalfNormal('sigma_a', sd=10)
    s_bg = HalfNormal('sigma_b', sd=10)

    #個人パラメータ
    a = Lognormal('a', mu=a_0,sd=s_ag , shape=[n_patient,1])
    b = Lognormal('b', mu=b_0, sd=s_bg , shape=[n_patient,1])
    
    #個人分散
    s_Y = HalfNormal('sigma_Y', sd=10)
    
    #likelihood 
    mu = 1-tt.exp( -tt.dot(b,times) )
    mu = a*mu
    Y_obs = Normal('Y_obs', mu=mu, sd=s_Y, observed=Y_nt)
    
    #サンプリング
    trace = sample(100)
    summary(trace)