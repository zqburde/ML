'''
网格搜索 可以先固定一个参数,优化后继续调整 
第一步：使用一个较大的learning_rate来确定大致的n_estimators
第二步： max_depth 和 min_child_weight 对最终结果有很大的影响 'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2),先大范围地粗调参数，然后再小范围地微调。 
第三步：gamma参数调优 'gamma':[i/10.0 for i in range(0,5)] 
第四步：调整subsample 和 colsample_bytree 参数 'subsample':[i/100.0 for i in range(75,90,5)], 'colsample_bytree':[i/100.0 for i in range(75,90,5)] 
第五步：正则化参数调优 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]先大范围确定大致值，后根据大致值小范围调优'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
       'reg_lambda' 同理
第六步：降低学习速率 learning_rate =0.01,最后优化模型
        确定学习速率和tree_based 给个常见初始值,根据是否类别不平衡调节max_depth,min_child_weight,起始值在4-6之间都是不错的选择; 
        min_child_weight比较小的值解决极不平衡的分类问题eg:1;
        subsample = 0.8, colsample_bytree = 0.8: 这个是最常见的初始值了 
        scale_pos_weight = 1: 这个值是因为类别十分不平衡。
'''

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from pandas import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

x, y = make_hastie_10_2(random_state=0)
y = DataFrame(y)   #转换成dataframe格式将标签－1改为0
y.columns = {"label"}
label = {-1:0, 1:1}
y.label = y.label.map(label)  #把－1变为0
y = y.as_matrix().flat[:]     #将dataframe转回ndarray
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


#使用一个较大的learning_rate来确定大致的n_estimators
def modelfit(alg, x_train, y_train, x_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        xgtest = xgb.DMatrix(x_test, label=y_test)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_xgb_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='auc')

    # Predict test data
    dtest_predictions = alg.predict(x_test)
    dtest_predprob = alg.predict_proba(x_test)[:, 1]

    # print model report
    print("Accuracy_1 (Test): %.4g" % metrics.accuracy_score(y_test, dtest_predictions))
    print("AUC Score_1 (Test): %f" % metrics.roc_auc_score(y_test, dtest_predprob))

    plt.figure(figsize=(8, 4))
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()



xgb1 = XGBClassifier(
        learning_rate =0.4,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb1, x_train, y_train, x_test, y_test)
#{'n_estimators': 289}
#Accuracy_1 (Test): 0.94
#AUC Score_1 (Test): 0.988415


#Grid seach on subsample and max_features
param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.4, n_estimators=289, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_train, y_train)
print(gsearch1.best_params_, '\n', gsearch1.best_score_, '\n', gsearch1.best_estimator_, '\n')
#{'max_depth': 3, 'min_child_weight': 1}
# 0.990558501037


#Grid seach on subsample and max_features
param_test2 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.4, n_estimators=289, max_depth=3,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(x_train, y_train)
print(gsearch2.best_params_, '\n', gsearch2.best_score_, '\n', gsearch2.best_estimator_, '\n')
#{'gamma': 0.4}
#0.991699838822


xgb2 = XGBClassifier(
        learning_rate =0.4,
        n_estimators=1000,
        max_depth=3,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb2, x_train, y_train, x_test, y_test)
#441
#Accuracy_1 (Test): 0.9513
#AUC Score_1 (Test): 0.992634


#Grid seach on subsample and max_features
param_test3 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.4, n_estimators=441, max_depth=3,
                                        min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(x_train, y_train)
print(gsearch3.best_params_, '\n', gsearch3.best_score_, '\n', gsearch3.best_estimator_, '\n')
#{'subsample': 0.8, 'colsample_bytree': 0.9}
#0.992071767475


#Grid seach on subsample and max_features
param_test4 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.4, n_estimators=441, max_depth=3,
                                        min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.9,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(x_train, y_train)
print(gsearch4.best_params_, '\n', gsearch4.best_score_, '\n', gsearch4.best_estimator_, '\n')
#{'reg_alpha': 1e-05}
#0.992075623268


param_test5 = {
    'reg_alpha':[0, 0.00001, 0.0001]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.4, n_estimators=441, max_depth=3,
                                        min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.9,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(x_train, y_train)
print(gsearch5.best_params_, '\n', gsearch5.best_score_, '\n', gsearch5.best_estimator_, '\n')
#{'reg_alpha': 1e-05} 
#0.992075623268 


xgb3 = XGBClassifier(
        learning_rate =0.4,
        n_estimators=1000,
        max_depth=3,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=1e-05,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb3, x_train, y_train, x_test, y_test)
#499
#Accuracy_1 (Test): 0.9502
#AUC Score_1 (Test): 0.992412 与xgb2相比精度降低了


xgb4 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=2000,
        max_depth=3,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=1e-05,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb4, x_train, y_train, x_test, y_test)
#1406
#Accuracy_1 (Test): 0.9523
#AUC Score_1 (Test): 0.993056


xgb5 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=2000,
        max_depth=3,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb5, x_train, y_train, x_test, y_test)
#893
#Accuracy_1 (Test): 0.95
#AUC Score_1 (Test): 0.992688


xgb6 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=5000,
        max_depth=3,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=1e-05,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb6, x_train, y_train, x_test, y_test)
#5000
#Accuracy_1 (Test): 0.9452
#AUC Score_1 (Test): 0.991543   降低学习速率和增大n_eatimators并没有提高精度
#综上 xgb4效果最好

