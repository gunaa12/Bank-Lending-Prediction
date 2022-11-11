
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


# getting the data
df = pd.read_csv("lending_train.csv")
df = df.drop(columns=['ID', 'race'])
df


# In[4]:


# normalizing columns
columns_to_norm = ['requested_amnt', 'annual_income', 'debt_to_income_ratio', 'fico_score_range_low', 'fico_score_range_high', 'revolving_balance', 'total_revolving_limit']
for col in columns_to_norm:
    df[col] = (df[col] - df[col].mean()) / df[col].std()


# In[ ]:


# one-hot encoding necessary columns
cols_to_one_hot_encode = ['type_of_application', 'reason_for_loan', 'extended_reason', 'employment_verified']
new_cols = pd.get_dummies(df, columns=cols_to_one_hot_encode)
df = df.join(new_cols)
df = df.drop(columns=cols_to_one_hot_encode)


# In[ ]:


df

