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

df_main = pd.read_csv("data/data-attendance-4-2.txt")
df_student =  pd.read_csv("data/data-attendance-4-1.txt")

n = df_main.shape[0] #出欠リストのサイズ
n_student = df_student.shape[0] #学生数
n_course = len(df_main["CourseID"].unique())
n_weather = 3

#天気変数の変換
df_main.loc[(df_main.Weather == "A" ),"Weather"] = 0
df_main.loc[(df_main.Weather == "B"),"Weather"] = 0.2
df_main.loc[(df_main.Weather == "C" ),"Weather"] = 1

#メインデータフレームにバイトの有無をコピー
df_main["part_time"] = df_main["Y"]
for i in range(n_student):
    df_main.loc[(df_main.PersonID == i+1 ),"part_time"] = df_student["A"].values[i]

#メインデータフレームにScoreをコピー
df_main["Score"] = df_main["Y"]
for i in range(n_student):
    df_main.loc[(df_main.PersonID == i+1 ),"Score"] = df_student["Score"].values[i]

P_ID = df_main.values[:,0].astype(np.int32) -1
C_ID = df_main.values[:,1].astype(np.int32) -1
Weather = df_main.values[:,2].astype(np.float32)
Y = df_main.values[:,3].astype(np.float32)
A =  df_main.values[:,4].astype(np.float32)
Score = df_main.values[:,5].astype(np.float32)


#shape = クラスの数の確率変数に、クラスの値を取るデータ数次元のベクトルを入れる操作がありますが
#その詳細な説明は(https://pymc-devs.github.io/pymc3/notebooks/GLM-hierarchical.html)参照

basic_model = Model()

with basic_model:
    #グローバルな変数
    b_1 = Normal('b_1',mu=0, sd=10)
    b_2 = Normal('b_2',mu=0, sd=10)
    b_3 = Normal('b_3',mu=0, sd=10)
    b_4 = Normal('b_4',mu=0, sd=10)
    
    #天気依存変数
    x_weather = b_4*Weather
    
    #科目依存変数
    s_c = HalfNormal('s_c',sd=10)
    x_course = Normal('x_course', mu=0, sd=s_c, shape=n_course)
    x_course = x_course[C_ID]
    
    #学生依存変数
    s_p = HalfNormal('s_p',sd=10)
    b_student_variance = Normal('b_student',mu=0, sd=s_p, shape=n_student)
    x_student = tt.dot(b_2,A) + tt.dot(b_3,Score) + b_student_variance[P_ID]
    
    #likelihood (NormalをBernoulliにするとNUTSにできないので、分散がほとんどない正規分布で対処.)
    x = b_1 + x_student + x_course + x_weather
    Y_obs = Normal('Y_obs', mu=tt.nnet.sigmoid(x), sd=0.00001, observed=Y)
    
    #サンプリング
    trace = sample(1000)
    summary(trace)