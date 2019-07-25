
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
df = pd.read_csv(r"data\happiness_train_complete.csv", encoding="GB2312")
#打乱样本顺序，随机分布
df = df.sample(frac=1, replace=False, random_state=1)
print(df)
df.reset_index(inplace=True)
print(df)
df = df[df["happiness"] > 0]
print(df)
Y = df["happiness"]
print(Y)

