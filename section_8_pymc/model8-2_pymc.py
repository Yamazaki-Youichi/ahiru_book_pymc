# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary


df = pd.read_csv("data-salary-2.txt")
X_data = df.values[:,0]
Y_data = df.values[:,1]
Class_data  = df.values[:,2]-1
n_Class = len(df["KID"].unique())


basic_model = Model()

with basic_model:
    a = Normal('a', mu=0, sd=10, shape=n_Class)
    b = Normal('b', mu=0, sd=10, shape=n_Class)
    epsilon = HalfNormal('sigma', sd=1)

    #likelihood 
    mu = a[Class_data] + b[Class_data]*X_data
    Y_obs = Normal('Y_obs', mu=mu, sd=epsilon, observed=Y_data)
    trace = sample(2000)
    summary(trace)