import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel('[전체]대선후보_유튜브_머신러닝_211129-220308_양선욱220429.xlsx', sheet_name='SNS2')

X = df[['Lee_tw_ta','Lee_tw_sp','Lee_tw_sn','Lee_tw_sg','Lee_bl_ta','Lee_bl_sp','Lee_bl_sn','Lee_bl_sg',
'Lee_co_ta','Lee_co_sp','Lee_co_sn','Lee_co_sg','Lee_is_ta','Lee_is_sp','Lee_is_sn','Lee_is_sg',
'Yoon_tw_ta','Yoon_tw_sp','Yoon_tw_sn','Yoon_tw_sg','Yoon_bl_ta','Yoon_bl_sp','Yoon_bl_sn','Yoon_bl_sg',
'Yoon_co_ta','Yoon_co_sp','Yoon_co_sn','Yoon_co_sg','Yoon_is_ta','Yoon_is_sp','Yoon_is_sn','Yoon_is_sg']]

# Y_num = df[['Lee_true','Yoon_true']]
# Y_num = df['Lee_true']
Y_num = df['Yoon_true']



X_train, X_test, Y_num_train, Y_num_test = train_test_split(X[:92], Y_num[:92], test_size=0.3, random_state=0)

test_input = X[92:]
test_answer_num = Y_num[92:]

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
test_input = scaler.transform(test_input)

np.random.seed(0)

#표준선형회귀분석 수치예측
# from sklearn.linear_model import LinearRegression
# model_num = LinearRegression().fit(X_train, Y_num_train)

#릿지 선형 회귀모형
# data = {'RMSE':[],'alpha':[],'pred':[], 'pred_RMSE':[]} 
# df_test = pd.DataFrame(data)
# from sklearn.linear_model import Ridge
# alpha = [0, 0.01, 0.1, 1, 10, 100]
# for a in alpha : 
#     model_num = Ridge(alpha=a,random_state=0).fit(X_train, Y_num_train)
#     Y_pred = model_num.predict(X_test)
  
#     from sklearn.metrics import mean_squared_error
#     from math import sqrt
#     rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
#     kkam_pred = model_num.predict(test_input)
#     rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))

#     newdata = {'RMSE':rmse,'alpha':a,'pred':kkam_pred, 'pred_RMSE':rmse_kkam}
#     df_test = df_test.append(newdata, ignore_index=True) 
# print(df_test.sort_values('RMSE'))
# print(df_test['pred'].loc[df_test.sort_values('RMSE').index[0]])


# KNN
# data = {'RMSE':[],'k':[],'pred':[], 'pred_RMSE':[]} 
# df_test = pd.DataFrame(data)
# from sklearn.neighbors import KNeighborsRegressor
# neighbors = [3,5,7,9,11,13,15]
# for n in neighbors : 
#     model_num = KNeighborsRegressor(n_neighbors= n, p=2)
#     model_num.fit(X_train, Y_num_train)
#     Y_pred = model_num.predict(X_test)

#     from sklearn.metrics import mean_squared_error
#     from math import sqrt
#     rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
#     kkam_pred = model_num.predict(test_input)
#     rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))

#     newdata = {'RMSE':rmse,'k':n,'pred':kkam_pred, 'pred_RMSE':rmse_kkam}
#     df_test = df_test.append(newdata, ignore_index=True) 
# print(df_test.sort_values('RMSE'))
# print(df_test['pred'].loc[df_test.sort_values('RMSE').index[0]])

#의사결정나무
# data = {'RMSE':[],'max_depth':[],'pred':[], 'pred_RMSE':[]} 
# df_test = pd.DataFrame(data)
# from sklearn.tree import DecisionTreeRegressor
# depth = [4,5,6,7,8,9,10,15]
# for depth in depth : 
#     model_num = DecisionTreeRegressor(random_state=0, max_depth=depth)
#     model_num.fit(X_train, Y_num_train)
#     Y_pred = model_num.predict(X_test)

#     from sklearn.metrics import mean_squared_error
#     from math import sqrt
#     rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))

#     kkam_pred = model_num.predict(test_input)
#     rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))

#     newdata = {'RMSE':rmse,'max_depth':depth,'pred':kkam_pred, 'pred_RMSE':rmse_kkam}
#     df_test = df_test.append(newdata, ignore_index=True) 
# print(df_test.sort_values('RMSE'))
# print(df_test['pred'].loc[df_test.sort_values('RMSE').index[0]])


#모델 SVR
# data = {'RMSE':[],'C':[],'epsilon':[], 'pred':[], 'pred_RMSE':[]} 
# df_test = pd.DataFrame(data)
# from sklearn.svm import LinearSVR
# Cs = [5,10,20]
# epsilons = [10,5,1]
# for c in Cs: 
#     for epsilon in epsilons : 
#         epsilon = epsilon/100
#         model_num = LinearSVR(C = c, epsilon=epsilon, random_state=0)
#         model_num.fit(X_train, Y_num_train)
#         Y_pred = model_num.predict(X_test)

#         from sklearn.metrics import mean_squared_error
#         from math import sqrt
#         rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))

#         kkam_pred = model_num.predict(test_input)
#         rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))

#         newdata = {'RMSE':rmse,'C':c,'epsilon':epsilon,'pred':kkam_pred, 'pred_RMSE':rmse_kkam}
#         df_test = df_test.append(newdata, ignore_index=True) 

# print(df_test.sort_values('RMSE'))
# print(df_test['pred'].loc[df_test.sort_values('RMSE').index[0]])


#모델 ANN
# data = {'RMSE':[],'alpha':[],'max_iter':[],'hidden_layer_sizes':[],'activation':[], 'pred':[], 'pred_RMSE':[]} 
# df_test = pd.DataFrame(data)
# from sklearn.neural_network import MLPRegressor
# Alphas = [0.001,0.01,0.1,0.5,1]
# Activations = ['relu','logistic','tanh','identity']
# Hidden_layers = [[50,50],[100,100]]
# for Alpha in Alphas :
#     for Activation in Activations : 
#         for Hidden_layer in Hidden_layers:
#             model_num = MLPRegressor(random_state=0, alpha=Alpha, max_iter=1000, hidden_layer_sizes=Hidden_layer, activation=Activation)
#             model_num.fit(X_train, Y_num_train)
#             Y_pred = model_num.predict(X_test)
 
#             from sklearn.metrics import mean_squared_error
#             from math import sqrt
#             rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))

#             kkam_pred = model_num.predict(test_input)
#             rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))

#             newdata = {'RMSE':rmse,'alpha':Alpha,'max_iter':'1000','hidden_layer_sizes':Hidden_layer,'activation':Activation,'pred':kkam_pred, 'pred_RMSE':rmse_kkam}
#             df_test = df_test.append(newdata, ignore_index=True) 

# print(df_test.sort_values('RMSE').head())
# print(df_test['pred'].loc[df_test.sort_values('RMSE').index[0]])


#보팅앙상블
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
knn = KNeighborsRegressor(n_neighbors= 11, p=2)
Dtree = DecisionTreeRegressor(random_state=0, max_depth=9)
ANN = MLPRegressor(random_state=0, alpha=0.010, max_iter=1000, hidden_layer_sizes=[50,50], activation='identity')
SVR = LinearSVR(C = 20, epsilon=0.1, random_state=0)
Ridge = Ridge(alpha=10,random_state=0).fit(X_train, Y_num_train)
model_num = VotingRegressor(estimators=[('Ridge',Ridge),('KNN',knn),('ANN',ANN)])
model_num.fit(X_train, Y_num_train)


Y_pred = model_num.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
print('모델 RMSE : ', rmse)

kkam_pred = model_num.predict(test_input)
print('깜깜이 기간 예측값',kkam_pred)

rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))
print('깜깜이 RMSE : ',rmse_kkam)
