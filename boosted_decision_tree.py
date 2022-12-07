
# coding: utf-8

# In[11]:


# imports
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import torch.nn as nn


# In[13]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(42, 84)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# In[46]:


# # boosted decision tree
# class BoostedDecisionTree:
#     def __init__(self, num_of_trees, max_depth=1):
#         self.trees = []
#         self.weights = []
#         self.num_of_trees = num_of_trees
#         self.max_depth = max_depth

#     # try and see how random forest compares
#     # try using class weights here
#     # try gini and see training time change
#     def fit(self, X, y, init_weights):
#         N = X.shape[0]
#         w = init_weights

#         for m in range(self.num_of_trees):
#             tree = DecisionTreeClassifier(max_depth=self.max_depth, criterion="gini")

#             norm_w = np.exp(w)
#             norm_w /= norm_w.sum()
            
#             tree.fit(X, y, sample_weight=norm_w)
#             yhat = tree.predict(X)

#             eps = norm_w.dot(yhat != y) + 1e-20
#             alpha = (np.log(1 - eps) - np.log(eps)) / 2

#             w = w - alpha * y * yhat

#             self.trees.append(tree)
#             self.weights.append(alpha)

#     def predict(self,X):
#         # try fixing this as current predictions arounf 48-52% are overwritten
#         y = np.zeros(X.shape[0])
#         for (tree, alpha) in zip(self.trees, self.weights):
#             y = y + alpha * tree.predict(X)
#         y = y - y.min()
#         y = [round(num) for num in y]
#         return y


# In[3]:


# getting the data
df = pd.read_csv('data/train_preprocessed_data.csv')
pd.set_option('max_columns', None)
df['loan_paid'].value_counts()


# In[4]:


# setting up data
y_train = df['loan_paid']
x_train = df.drop(columns=['loan_paid']).to_numpy()


# In[5]:


# fixing numpy conversion issues
print(np.where(np.isinf(x_train)))
x_train[np.isinf(x_train)] = 0
np.where(np.isinf(x_train))


# In[8]:


# logistic regression model
# lg = LogisticRegression(random_state=0).fit(x_train, y_train)


# In[106]:


# # random forest model
# rfc = RandomForestRegressor(n_estimators=500, max_depth=1)
# rfc.fit(x_train, y_train)


# In[27]:


# # getting stored cluster_count and k_means model
# with open('models/k_means.pickle', 'rb') as handle:
#     k_means = pickle.load(handle)
    
# with open('data/cluster_count.pickle', 'rb') as handle:
#     cluster_count = pickle.load(handle)


# In[28]:


# # generating init_weights
# init_weights = []
# total_count = sum(cluster_count)
# cluster_pred = k_means.predict(x_train)

# for pred in cluster_pred:
#     init_weights.append(cluster_count[pred] / total_count)

# init_weights = np.array(init_weights)
# init_weights = init_weights / sum(init_weights)
# init_weights


# In[29]:


# # training the data
# # maybe overfitting? reduce # of trees necessary
# model = BoostedDecisionTree(250, 1)
# model.fit(x_train, y_train, init_weights)


# In[79]:


# getting validation input
x_val = pd.read_csv('data/predict_preprocessed_data.csv')
x_val_id = x_val['ID']
x_val = x_val.drop(columns=['ID'])


# In[81]:


# converting to np array
x_val = x_val.to_numpy()
print(np.where(np.isinf(x_val)))
x_val[np.isinf(x_val)] = 0
np.where(np.isinf(x_val))


# In[1]:


# # predicting using logistic regression
# pred_probs = lg.predict(x_val)
# pred_probs


# In[104]:


# # predicting using random forest
# pred_probs = rfc.predict(x_val)
# zero = min(pred_probs)
# pred = []
# for prob in pred_probs:
#     if (prob == zero):
#         pred.append(0)
#     else:
#         pred.append(1)
# pred


# In[ ]:


network = Net()
network.fit(x_train, y_train)
pred = network.predict(x_val)
print(pred)


# In[105]:


# creating submission
submission = pd.DataFrame()
submission['ID'] = x_val_id
submission['loan_paid'] = pred
submission.to_csv('submissions/pred.csv', index=None)

