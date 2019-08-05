import pandas as pd
import numpy as np

train_data = pd.read_csv("./data/happiness_train_abbr.csv", index_col='id')
test_data = pd.read_csv("./data/happiness_test_abbr.csv", index_col='id')
list = ['survey_time', 'work_status', 'work_yr', 'work_type', 'work_manage']
for i in list:
    del train_data[i]
    del test_data[i]
    test_index = test_data.index

# 主成分分析，数据降维
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

def data_deal(train_data):
    for i in train_data.columns:
        for j in train_data.index:
            if np.isnan(train_data[i][j]):
                train_data[i][j] = 0
                print(train_data[i][j])
    pca = PCA(n_components=8)#选取8个主成分
    re = pca.fit_transform(train_data)
    return re
train = pd.read_csv(r"./data/happiness_train_abbr.csv")
test = pd.read_csv(r"./data/happiness_test_abbr.csv")

train_data_x = train.drop(columns=['id','happiness','work_status','work_yr','work_type','work_manage','survey_time','family_income'])
train_data_x=data_deal(train_data_x)

train_data_y = train["happiness"]
train_data_y = train_data_y.map(lambda x:3 if x== -8 else x)

test_data = test.drop(columns=['id','work_status','work_yr','work_type','work_manage','survey_time','family_income'])
test_data=data_deal(test_data)

# 建立神经网络模型，进行幸福预测
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import TensorBoard

model = Sequential()
model.add(Dense(input_dim=8, output_dim=16))
model.add(Activation('relu'))
model.add(Dense(input_dim=16,output_dim=1))
# model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer='Adagrad')
model.fit(train_data_x, train_data_y, nb_epoch=1400, batch_size=20, callbacks=[TensorBoard(log_dir='mytensorboard')])#利用TensorBoard记录训练曲线
print(model.predict(test_data))
r = pd.DataFrame(model.predict(test_data), index=test_index)
r.to_csv("forcast.csv")
