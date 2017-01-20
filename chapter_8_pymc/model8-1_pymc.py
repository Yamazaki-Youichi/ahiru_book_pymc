# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot
from pymc3 import summary


df = pd.read_csv("data-salary-2.txt")
X_data = df.values[:,0]
Y_data = df.values[:,1]

basic_model = Model()

with basic_model:
    a = Normal('a', mu=0, sd=10)
    b = Normal('b', mu=0, sd=10)
    epsilon = HalfNormal('sigma', sd=1)

    mu = a + b*X_data
    Y_obs = Normal('Y_obs', mu=mu, sd=epsilon, observed=Y_data)

    trace = sample(2000)
    summary(trace)