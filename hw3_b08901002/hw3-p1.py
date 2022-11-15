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


# In[3]:


print(indexes)


# In[4]:


print(indexes.iloc[4])


# In[5]:


print(x.shape)


# In[6]:


print(x)


# In[7]:


print(y.shape)


# In[8]:


print(y)


# # Feature ranking : One-by-one Feature Selection

# # Use Similarity-Based: Fisher Score (without normalization)

# In[9]:


def get_ui(x,i): #data x and index i, ui is the mean of feature i
    target = x[:,i]
    return np.mean(target)


# In[10]:


def get_uij_and_varij(x,i,j):
    target = []
    for k in range(62):
        if y[k] == j:
            target.append(x[k,i])
            
    uij = np.mean(target)
    varij = np.var(target)
    return uij,varij


# In[11]:


def insert(idx,idx_score,i,i_score):
    output_idx = idx
    output_idx_score = idx_score
    for k in range(len(output_idx)):
        if i_score >= output_idx_score[k]:
            output_idx.insert(k,i)
            output_idx_score.insert(k,i_score)
            break
    return output_idx,output_idx_score


# In[12]:


# TODO: Design your score function for feature selection
# ranking_idx = np.linspace(0,1999,2000,dtype=int)
# random.shuffle(ranking_idx)
ranking_idx = [-1]
ranking_idx_score = [-1]
nij = [62-np.count_nonzero(y),np.count_nonzero(y)]
for i in range(2000): #pick ith feature
#     print(i)
    up = 0
    down = 0
    for j in range(2): #class j
        nj = nij[j] 
        uij,varij = get_uij_and_varij(x,i,j)
        ui = get_ui(x,i)
        up += nj*((uij-ui)**2)
        down += nj*varij
    fisher_score = up/down
    ranking_idx,ranking_idx_score = insert(ranking_idx,ranking_idx_score,i,fisher_score)
# TODO: To use the provided evaluation sample code, you need to generate ranking_idx, which is the sorted index of feature


# # Feature evaluation

# In[13]:


# Use a simple dicision tree with 5-fold validation to evaluate the feature selection result.
# You can try other classifier and hyperparameter.
score_history = []
for m in range(1, 2001, 1): # m = 5,10,15,...,2000 -> m = 1,2,3,...,2000
    # Select Top m feature
    x_subset = x[:, ranking_idx[:m]]

    # Build random forest
    clf = DecisionTreeClassifier(random_state=0)
#     clf = SVC(kernel='rbf', random_state=0) #build SVM

    # Calculate validation score
    scores = cross_val_score(clf, x_subset, y, cv=5)

    # Save the score calculated with m feature
    score_history.append(scores.mean())

# Report best accuracy.
print(f"Max of Decision Tree: {max(score_history)}")
# print(f"Number of features: {np.argmax(score_history)*5+5}")
print(f"Number of features: {np.argmax(score_history)}")


# # Visualization

# In[14]:


plt.plot(range(1, 2001, 1), score_history, c='blue')
plt.title('Original')
plt.xlabel('Number of features')
plt.ylabel('Cross-validation score')
plt.legend(['Decision Tree'])
plt.savefig('1-3_result.png')


# # Record (Max of Decision Tree / Number of features)

# completely_random : 0.8538461538461538 / 460

# fisher_score = 0.8705128205128204 / 35

# fisher_score(skip=1) = 0.9038461538461539 / 67

# In[15]:


np.var(x[:,1])


# In[16]:


np.count_nonzero(y)


# In[17]:


# print(ranking_idx)


# In[18]:


print(ranking_idx_score)


# # Validate mutiple features case

# pick index = 248 , 764 , 492 features

# In[19]:


import numpy as np
z1 = np.reshape(x[:,248].T,(1,62))
z2 = np.reshape(x[:,764].T,(1,62))
z3 = np.reshape(x[:,492].T,(1,62))
Z = np.concatenate((z1,z2,z3))
print(z1.shape)
print(Z.shape)


# In[20]:


np.reshape(Z[:,61],(3,1))


# # get SB

# In[21]:


def cal_uk(Z,k):
    for i in range(62):
        if int(y[i]) == k:
#             print('dfb')
            try:
                target = np.concatenate((target,np.reshape(Z[:,i],(3,1))),axis=1)
            except:
                target = np.reshape(Z[:,i],(3,1))
#     print(target.shape)
#     print(np.average(target,axis=1).shape)
    
    return np.average(target,axis=1)


# In[22]:


u0 = np.reshape(cal_uk(Z,0),(3,1))
u1 = np.reshape(cal_uk(Z,1),(3,1))
u = nij[0]*u0 + nij[1]*u1
SB = np.zeros((3,3))
for k in range(2):
    nk = nij[k]
    if k==0:
        uk=u0
    else:
        uk=u1
    SB += nk* np.matmul((uk-u),(uk-u).T)
# print(SB)


# # get St

# In[23]:


St = np.zeros((3,3))
for i in range(62):
    zi = np.reshape(Z[:,i],(3,1))
    St += np.matmul((zi-u),(zi-u).T)


# # get fisher score

# In[24]:


score = 0
A = np.matmul(SB,np.linalg.inv(St+1e-2*np.identity(3)))
for i in range(3):
    score += A[i,i]


# In[25]:


print(score)


# In[26]:


sum([0.6635444906271158,0.5524959834149036, 0.5335986349543933])


# # 結論: 故不是score直接相加

# In[27]:


a = [2 ,3]
a[:0]


# In[28]:


for m in range(1, 2001, 1):
    print(m)


# In[29]:


for m in range(5, 2001, 5):
    print(m)


# In[30]:


np.argmax(score_history)


# In[31]:


# print(score_history)
print(score_history[67])


# In[ ]:




