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


df = pd.read_excel('[전체]대선후보_유튜브_머신러닝_211129-220308_양선욱220429.xlsx', sheet_name='유튜브+트렌드')

X = df[['Lee_sp','Lee_sn','Lee_sg','Lee_vn','Lee_hn','Lee_ln','Lee_cn','Yoon_sp','Yoon_sn','Yoon_sg','Yoon_vn','Yoon_hn','Yoon_ln','Yoon_cn',
'Lee_gtw','Lee_gty','Lee_ndl','Lee_kdt','Yoon_gtw','Yoon_gty','Yoon_ndl','Yoon_kdt']]
# Y_num = df[['Lee_true','Yoon_true']]
# Y_num = df['Lee_true']
Y_num = df['Yoon_true']
# Y_cat = df['Lee_Yoon_per']


X_train, X_test, Y_num_train, Y_num_test = train_test_split(X[:92], Y_num[:92], test_size=0.3, random_state=0)
# X_train, X_test, Y_cat_train, Y_cat_test = train_test_split(X[:93], Y_cat[:93], test_size=0.3, random_state=0)

test_input = X[92:]
test_answer_num = Y_num[92:]
# test_answer_cat = Y_cat[92:]

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
test_input = scaler.transform(test_input)

np.random.seed(0)

#종속변수 정규성 확인
# test_stat, p_value = stats.shapiro(Y_num[:92])
# print('종속변수변수 - 통계값 : ',round(test_stat,3),' p_value : ',round(p_value,3))
# plt.title('종속 변수의 히스토그램')
# sns.distplot(Y_num[:93])
# plt.show()
# plt.clf()
# log_data = np.log1p(Y_num[:93])
# plt.title('종속 변수의 로그히스토그램')
# sns.distplot(log_data)
# plt.show()
# plt.clf()


#표준선형회귀분석 수치예측
# from sklearn.linear_model import LinearRegression
# model_num = LinearRegression().fit(X_train, Y_num_train)

#릿지 선형 회귀모형
# from sklearn.linear_model import Ridge
# alpha = [0, 0.01, 0.1, 1, 10, 100]
# for a in alpha : 
#     model_num = Ridge(alpha=a,random_state=0).fit(X_train, Y_num_train)
#     Y_pred = model_num.predict(X_test)
#     train_num_score = model_num.score(X_train, Y_num_train)
#     test_num_score = model_num.score(X_test, Y_num_test)
#     print('alpha : ',a)
#     print('학습용 데이터 세트 MSE : ', train_num_score)
#     print('평가용 데이터 세트 MSE : ', test_num_score)

#     from sklearn.metrics import mean_squared_error
#     from math import sqrt
#     rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
#     print('모델 RMSE : ', rmse)

#     kkam_pred = model_num.predict(test_input)
#     print('깜깜이 기간 예측값',kkam_pred)

#     rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))
#     print('깜깜이 RMSE : ',rmse_kkam)

# KNN
# from sklearn.neighbors import KNeighborsRegressor
# neighbors = [3,4,5,6,7,8,9,10,15]
# for n in neighbors : 
#     model_num = KNeighborsRegressor(n_neighbors= n, p=2)
#     model_num.fit(X_train, Y_num_train)
#     Y_pred = model_num.predict(X_test)
#     train_num_score = model_num.score(X_train, Y_num_train)
#     test_num_score = model_num.score(X_test, Y_num_test)
#     print('k : ',n)
#     print('학습용 데이터 세트 MSE : ', train_num_score)
#     print('평가용 데이터 세트 MSE : ', test_num_score)

#     from sklearn.metrics import mean_squared_error
#     from math import sqrt
#     rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
#     print('모델 RMSE : ', rmse)

#     kkam_pred = model_num.predict(test_input)
#     print('깜깜이 기간 예측값',kkam_pred)

#     rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))
#     print('깜깜이 RMSE : ',rmse_kkam)


#의사결정나무
# from sklearn.tree import DecisionTreeRegressor
# depth = [4,5,6,7,8,9,10,15]
# for depth in depth : 
#     model_num = DecisionTreeRegressor(random_state=0, max_depth=depth)
#     model_num.fit(X_train, Y_num_train)
#     Y_pred = model_num.predict(X_test)
#     train_num_score = model_num.score(X_train, Y_num_train)
#     test_num_score = model_num.score(X_test, Y_num_test)
#     print('depth : ',depth)
#     print('학습용 데이터 세트 MSE : ', train_num_score)
#     print('평가용 데이터 세트 MSE : ', test_num_score)

#     from sklearn.metrics import mean_squared_error
#     from math import sqrt
#     rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
#     print('모델 RMSE : ', rmse)

#     kkam_pred = model_num.predict(test_input)
#     print('깜깜이 기간 예측값',kkam_pred)

#     rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))
#     print('깜깜이 RMSE : ',rmse_kkam)



#모델 SVR
# from sklearn.svm import LinearSVR
# Cs = [5,10,20]
# epsilons = [10,5,1]
# for c in Cs: 
#     for epsilon in epsilons : 
#         epsilon = epsilon/100
#         model_num = LinearSVR(C = c, epsilon=epsilon, random_state=0)
#         model_num.fit(X_train, Y_num_train)
#         Y_pred = model_num.predict(X_test)
#         train_num_score = model_num.score(X_train, Y_num_train)
#         test_num_score = model_num.score(X_test, Y_num_test)
#         print('Cs : ',c)
#         print('epsilon : ', epsilon)
#         print('학습용 데이터 세트 MSE : ', train_num_score)
#         print('평가용 데이터 세트 MSE : ', test_num_score)

#         from sklearn.metrics import mean_squared_error
#         from math import sqrt
#         rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
#         print('모델 RMSE : ', rmse)

#         kkam_pred = model_num.predict(test_input)
#         print('깜깜이 기간 예측값',kkam_pred)

#         rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))
#         print('깜깜이 RMSE : ',rmse_kkam)





#모델 ANN
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
#             train_num_score = model_num.score(X_train, Y_num_train)
#             test_num_score = model_num.score(X_test, Y_num_test)
#             print('Alphas : ',Alpha)
#             print('Activations : ', Activation)
#             print('hiden layer : ', Hidden_layer)
#             print('학습용 데이터 세트 MSE : ', train_num_score)
#             print('평가용 데이터 세트 MSE : ', test_num_score)

#             from sklearn.metrics import mean_squared_error
#             from math import sqrt
#             rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
#             print('모델 RMSE : ', rmse)

#             kkam_pred = model_num.predict(test_input)
#             print('깜깜이 기간 예측값',kkam_pred)

#             rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))
#             print('깜깜이 RMSE : ',rmse_kkam)


#보팅앙상블
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
knn = KNeighborsRegressor(n_neighbors= 7, p=2)
Dtree = DecisionTreeRegressor(random_state=0, max_depth=9)
ANN = MLPRegressor(random_state=0, alpha=0.001, max_iter=1000, hidden_layer_sizes=[100,100], activation='identity')
model_num = VotingRegressor(estimators=[('knn',knn),('Dtree',Dtree),('ANN',ANN)])





model_num.fit(X_train, Y_num_train)
Y_pred = model_num.predict(X_test)
train_num_score = model_num.score(X_train, Y_num_train)
test_num_score = model_num.score(X_test, Y_num_test)
print('학습용 데이터 세트 MSE : ', train_num_score)
print('평가용 데이터 세트 MSE : ', test_num_score)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(Y_num_test, Y_pred))
print('모델 RMSE : ', rmse)

kkam_pred = model_num.predict(test_input)
print('깜깜이 기간 예측값',kkam_pred)

rmse_kkam = sqrt(mean_squared_error(test_answer_num, kkam_pred))
print('깜깜이 RMSE : ',rmse_kkam)

