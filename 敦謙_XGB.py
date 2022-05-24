import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt # ploting the data
import seaborn as sns # ploting the data
import math # calculation
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from matplotlib import pyplot
import pickle
data = pd.read_csv("train.csv")
data_dec = data.describe()
data.info()
#轉換'PageValues'超過0就=1; 在此情況下: (0,0; 6048) (1,1; 1025); (0,1; 775) (1,0; 252) 準確率約87%
# <思考1: 題目是否需拿掉此feature> <思考2: 針對Revenue_0沒有訂房，卻有PageValues的客戶進行分析>
data['PageValues_dummy'] = np.where(data['PageValues'] > 0, [1],data['PageValues'])
pd.crosstab(data.PageValues_dummy, data.Revenue)
#解法一: 只轉換PageValues，極端值不處理_RF_Xgboost兩種模型
df_new = data.drop(columns=['ID', 'Revenue'])
X = df_new.values
Y = data['Revenue'].values
X_train_org, X_test_org, y_train, y_test = train_test_split(X, Y, test_size=0.3)
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train_org.astype(np.float))
X_test = s_scaler.transform(X_test_org.astype(np.float))

def score(m, x_train, y_train, x_test, y_test, train=True):
    if train:
        pred=m.predict(x_train)
        print('Train Result:\n')
        print(f"Accuracy Score: {accuracy_score(y_train, pred)*100:.2f}%")
        print(f"Precision Score: {precision_score(y_train, pred)*100:.2f}%")
        print(f"Recall Score: {recall_score(y_train, pred)*100:.2f}%")
        print(f"F1 score: {f1_score(y_train, pred)*100:.2f}%")
        print(f"Confusion Matrix:\n {confusion_matrix(y_train, pred)}")
    elif train == False:
        pred=m.predict(x_test)
        print('Test Result:\n')
        print(f"Accuracy Score: {accuracy_score(y_test, pred)*100:.2f}%")
        print(f"Precision Score: {precision_score(y_test, pred)*100:.2f}%")
        print(f"Recall Score: {recall_score(y_test, pred)*100:.2f}%")
        print(f"F1 score: {f1_score(y_test, pred)*100:.2f}%")
        print(f"Confusion Matrix:\n {confusion_matrix(y_test, pred)}")

#都沒調，隨便找一個模型
# xgbc = XGBClassifier()
# print(xgbc)
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
#        max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
#        n_estimators=100, n_jobs=1, nthread=None,
#        objective='multi:softprob', random_state=0, reg_alpha=0,
#        reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
#        subsample=1, verbosity=1)
# xgbc.fit(X_train, y_train)
# score(xgbc, X_train, y_train, X_test, y_test, train=False)
# #Accuracy Score: 89.55%
# #Precision Score: 70.69%
# #Recall Score: 54.81%
# #F1 score: 61.75%
# #Confusion Matrix:
# # [[1971   85]
# # [ 169  205]]
#
#
# #Random CV調模型
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# learning_rate=[round(float(x),2) for x in np.linspace(start=0.01, stop=0.2, num=10)]
# colsample_bytree =[round(float(x),2) for x in np.linspace(start=0.1, stop=1, num=10)]
#
# random_grid = {'n_estimators': n_estimators,
#                'max_depth': max_depth,
#                'learning_rate': learning_rate,
#                'colsample_bytree': colsample_bytree}
# xg4 = XGBClassifier(random_state=42)
#
# #Random search of parameters, using 3 fold cross validation, search across 100 different combinations, and use all available cores
# xg_random = RandomizedSearchCV(estimator = xg4, param_distributions=random_grid,
#                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
#
# xg_random.fit(X_train,y_train)
# xg_random.best_params_
#
# xg1 = XGBClassifier(colsample_bytree= 0.4, learning_rate=0.05, max_depth= 10, n_estimators=800)
# xg1=xg1.fit(X_train, y_train)
# score(xg1, X_train, y_train, X_test, y_test, train=False)
#Accuracy Score: 89.22%
#Precision Score: 70.44%
#Recall Score: 51.60%
#F1 score: 59.57%
#Confusion Matrix:
# [[1975   81]
# [ 181  193]]



#Grid CV調模型
from sklearn.model_selection import GridSearchCV
params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}

xg2 = XGBClassifier(random_state=1)
clf = GridSearchCV(estimator=xg2,
                   param_grid=params,
                   scoring='neg_mean_squared_error',
                   verbose=1)
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
xg2 = XGBClassifier(colsample_bytree= 0.7, learning_rate=0.01, max_depth= 3, n_estimators=1000)
xg2=xg2.fit(X, Y)
score(xg2, X, Y,X, Y, train=False)
import pickle
filename = 'finalized_model.sav'
pickle.dump(xg2, open(filename, 'wb'))
#Accuracy Score: 90.16%
#Precision Score: 74.19%
#Recall Score: 55.35%
#F1 score: 63.40%
#Confusion Matrix:
 #[[1984   72]
 #[ 167  207]]


#test 測試
data_test = pd.read_csv("test.csv")
data_test['PageValues_dummy'] = np.where(data_test['PageValues'] > 0, [1],data_test['PageValues'])
df_test_new = data_test.drop(columns=['ID'])
df_test_new_x = df_test_new.values

df_test_new_x_input = s_scaler.fit_transform(df_test_new_x.astype(np.float))
loaded_model = pickle.load(open(filename, 'rb'))
y_pred_test = loaded_model.predict(df_test_new_x_input)
y_pred_test_prob = loaded_model.predict_proba(df_test_new_x_input)
ID_column = data_test["ID"]
y_pred_test = pd.Series(y_pred_test,name="Revenue_pred")
y_pred_test_prob = pd.DataFrame(y_pred_test_prob )
results_all = pd.concat([data_test, y_pred_test,y_pred_test_prob], axis=1)
results_all.to_excel('sample_sub.xls')

#用pageValue 當答案，預測的結果
pd.crosstab(results_all.Revenue_pred, results_all.PageValues_dummy)