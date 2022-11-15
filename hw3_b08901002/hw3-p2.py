#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import random


# # Load Data

# In[2]:


# TODO: Load data here.
indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None)
x = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None).to_numpy().T
y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
y = (y>0).astype(int).reshape(y.shape[0])


# # Feature ranking: Subset-Based Feature Selection 

# # Use PSO (particle swarm optimization)

# # use f(x) = a * (1 - P) + (1-a)* #_select/#_features

# x is 2000-D vector with all elements between 0-1

# If x[i] >0.5, then index = i feature is chosen, else the feature is discarded

# # Hyperparameter

# In[3]:


a = 0.9 #重視正確率(a大)或重視選少feature, a = 0~1
gamma = 1
w = 0.9/3 #0.1~10 調自己的速度貢獻
phi_p = 0.8/3 #0.1~10 調自己最佳貢獻
phi_g = 0.8/3 #0.1~10 調global最佳貢獻
lr = 3 #0.1~10 縮放最終速度
iteration = 20 #跑幾輪
S = 1000 #number of particles
k = 1 #Vmax[j] = k*((max(x[j])-min(x[j]))/2) , k = 0~1
w_decay = False


# # define cost function

# In[4]:


def cost_function(x,y,pi):
    idx = []
    for i in range(2000):
        if pi[i] > 0.5:
            idx.append(i)
    x_subset = x[:,idx]
    clf = DecisionTreeClassifier(random_state=0)
    scores = cross_val_score(clf, x_subset, y, cv=5)
    P = scores.mean()
    fx = a*(1 - P) + (1-a)*gamma*(len(idx))/(2000)
    return fx,P


# # Initialization

# In[5]:


ParticlePosition = np.random.uniform(low=0, high=1, size=(S,2000)) #Since there are 2000 features
ParticleBestPosition = ParticlePosition
ParticleBestCost = np.zeros((S,1))
ParticleVelocity = np.random.uniform(low=-1, high=1, size=(S,2000))
Vmax = k*0.5 #Vmax = k*(b_up-b_low)/2
g_cost = 65535 #global best value(least cost)
for i in range(S):
    cost,P = cost_function(x,y,ParticlePosition[i]) #smaller is better
    ParticleBestCost[i] = cost
    if cost < g_cost:
        g_cost = cost
        g = ParticlePosition[i]
        best_P = P


# # main algorithm

# In[6]:


# TODO: Design your score function for feature selection
P_record = []
for k in range(iteration):
    # w is linearly decay
#     print(k)
    if w_decay:
        wk = w - (w-0)*(k/iteration)
    else:
        wk = w
    for i in range(S): #for i-th particle
        for d in range(2000): #for each dimension
            rp = random.random()
            rg = random.random()
            ParticleVelocity[i][d] = wk*ParticleVelocity[i][d] +                                     phi_p*rp*(ParticleBestPosition[i][d] - ParticlePosition[i][d]) +                                     phi_g*rg*(g[d] - ParticlePosition[i][d])
            #keep the particle in space
            if ParticleVelocity[i][d] < -Vmax:
                ParticleVelocity[i][d] = -Vmax
            if ParticleVelocity[i][d] > Vmax:
                ParticleVelocity[i][d] = Vmax
            
            ParticlePosition[i][d] = ParticlePosition[i][d] + lr*ParticleVelocity[i][d]
            #keep the particle in space
            if ParticlePosition[i][d] < -1:
                ParticlePosition[i][d] = -1
            if ParticlePosition[i][d] > 1:
                ParticlePosition[i][d] = 1
            
        cost,P = cost_function(x,y,ParticlePosition[i])
        if cost < ParticleBestCost[i]:
            ParticleBestPosition[i] = ParticlePosition[i]
            ParticleBestCost[i] = cost
            if cost < g_cost:
                g = ParticlePosition[i]
                g_cost = cost
                best_P = P
    print("round",k,":",best_P)
    P_record.append(best_P)
# TODO: To use the provided evaluation sample code, you need to generate ranking_idx, which is the sorted index of feature


# In[7]:


# print(g)


# In[8]:


# print(P_record)
plt.plot(range(0, iteration, 1),P_record)


# In[9]:


pickingidx = []
for i in range(2000):
    if g[i]>0.5:
        pickingidx.append(i)


# In[10]:


print(best_P,'/',len(pickingidx))


# # baseline: 0.9038461538461539 / 67

# # Experiment Record

# a = 1
# w = 0.2
# phi_p = 0.4
# phi_g = 0.4
# lr = 0.05
# iteration = 40
# S = 100
# 
# -> 0.9012820512820513 / 991

# a = 0.7
# w = 0.2
# phi_p = 0.4
# phi_g = 0.4
# lr = 0.05
# iteration = 40
# S = 100
# 
# -> 0.8846153846153847 / 961

# a = 0.3
# w = 0.2
# phi_p = 0.4
# phi_g = 0.4
# lr = 0.05
# iteration = 40
# 
# #number of particles
# S = 100
# 
# ->0.8538461538461538/931
