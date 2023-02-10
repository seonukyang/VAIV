import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


df = pd.read_excel('[전체]대선후보_머신러닝_211129-220308_양선욱220429.xlsx', sheet_name='유튜브분석')

X = df[['Lee_sp','Lee_sn','Lee_sg','Lee_vn','Lee_hn','Lee_ln','Lee_cn','Yoon_sp','Yoon_sn','Yoon_sg','Yoon_vn','Yoon_hn','Yoon_ln','Yoon_cn']]
Y_num = df[['Lee_true','Yoon_true']]
# Y_num = df['Lee_true']
# Y_num = df['Yoon_true']
# Y_cat = df['Lee_Yoon_per']


X_train, X_test, Y_num_train, Y_num_test = train_test_split(X[:93], Y_num[:93], test_size=0.3, random_state=0)
# X_train, X_test, Y_cat_train, Y_cat_test = train_test_split(X[:93], Y_cat[:93], test_size=0.3, random_state=0)

test_input = X[93:]
test_answer_num = Y_num[93:]
# test_answer_cat = Y_cat[93:]

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
test_input = scaler.transform(test_input)

np.random.seed(0)
tf.random.set_seed(0)

model_num = keras.models.Sequential()
model_num.add(keras.layers.Dense(64, input_dim=14, activation='relu'))
model_num.add(keras.layers.Dense(64, activation='relu'))
model_num.add(keras.layers.Dense(64, activation='relu'))
model_num.add(keras.layers.Dense(64, activation='relu'))
model_num.add(keras.layers.Dense(64, activation='relu'))
model_num.add(keras.layers.Dense(2))

model_num.compile(losee='mse', optimizer='SGD')

model_num.fit(X_train, Y_num_train, epochs=20, batch_size=64, verbose=0)

Y_pred = model_num.predict(X_test, verbose=0)
train_num_score = model_num.score(X_train, Y_num_train)
test_num_score = model_num.score(X_test, Y_num_test)
print('학습용 데이터 세트 MSE : ', train_num_score)
print('평가용 데이터 세트 MSE : ', test_num_score)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
print('RMSE : ', rmse)

kkam_pred = model_num.predict(test_input)
print(kkam_pred)



