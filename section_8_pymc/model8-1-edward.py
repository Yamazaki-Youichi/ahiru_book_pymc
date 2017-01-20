# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv as csv
#import matplotlib.pyplot as plt
import tensorflow as tf
from edward.models import Normal, NormalWithSoftplusSigma,Uniform,Exponential, Empirical
import edward as ed
from tqdm import tqdm
import time

#グループ差無しの線形回帰

df = pd.read_csv("data-salary-2.txt")
X_data = np.reshape(df.values[:,0],(-1,1))
Y_data = df.values[:,1]

N = X_data.shape[0] #データ数

##モデル記述
#Y[n] ~ Y_base[n] + \epsilon
#Y_base[n] ~ a + bX[n]
#\epsilon ~ N(0,\sigma_Y)
X = tf.placeholder(tf.float32, [N, 1])
a = Normal(mu=tf.zeros([1]), sigma=tf.ones([1])*500)
b = Normal(mu=tf.zeros([1]), sigma=tf.ones([1])*500)
sigma = Uniform(a=tf.ones([1])*1, b=tf.ones([1])*10)
Y = Normal(mu=ed.dot(X, b) + a, sigma=sigma)

#データ
data = {X: X_data, Y: Y_data}

##推論(変分ベイズ)
#qa = Normal(mu=tf.Variable(tf.random_normal([1])),\
#	sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
#qb = Normal(mu=tf.Variable(tf.random_normal([1])),\
#	sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
#qsigma = Normal(mu=tf.Variable(tf.random_normal([1])),\
#	sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
#inference = ed.KLqp({a: qa, b: qb, sigma : qsigma}, data)
#inference.run(n_samples=1, n_iter=10000)

#HMC
qa = Empirical(params=tf.Variable(tf.random_normal([10000,1])))
qb = Empirical(params=tf.Variable(tf.random_normal([10000,1])))
qsigma = Empirical(params=tf.Variable(tf.random_normal([10000,1])))
inference = ed.HMC({a: qa, b: qb, sigma : qsigma}, data=data)
inference.run()

qa.sample(10).eval()
qb.sample(10000).eval()
