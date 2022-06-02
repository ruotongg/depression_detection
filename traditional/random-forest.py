import boto
import os
import sys
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


def prep_train_test(X_train, X_test):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))

    x_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
    x_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

    return x_train, x_test


def random_forest(X_train, y_train, X_test, y_test):

    rf0 = RandomForestClassifier(n_estimators=10, oob_score=True, max_features="auto", random_state=10)
    rf0.fit(X_train, y_train)
    print(rf0.oob_score_)

    param_test1 = {'n_estimators': list(range(10,80,10)),'max_depth': list(range(10,100,10))}
    gsearch = RandomizedSearchCV(estimator=RandomForestClassifier(oob_score=True, max_features='log2', random_state=10),
                                  param_distributions=param_test1, scoring='f1_micro', cv=5)

    gsearch.fit(X_train, y_train)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)

    rf1 = RandomForestClassifier(oob_score=True, max_features='auto', random_state=10, n_estimators=gsearch.best_params_['n_estimators'],max_depth=gsearch.best_params_['max_depth'])
    rf1.fit(X_train, y_train)
    y_predict = rf1.predict(X_test)

    print('准确率:', metrics.accuracy_score(y_test, y_predict))  # 预测准确率输出
    print('宏平均精确率:', metrics.precision_score(y_test, y_predict, average='macro'))  # 预测宏平均精确率输出
    print('微平均精确率:', metrics.precision_score(y_test, y_predict, average='micro'))  # 预测微平均精确率输出
    print('宏平均召回率:', metrics.recall_score(y_test, y_predict, average='macro'))  # 预测宏平均召回率输出
    print('平均F1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))  # 预测平均f1-score输出
    print('混淆矩阵输出:', metrics.confusion_matrix(y_test, y_predict))  # 混淆矩阵输出
    print('分类报告:', metrics.classification_report(y_test, y_predict))  # 分类报告输出


if __name__ == '__main__':

    path = '/mnt/xlancefs/home/rtg99/data/npz/'

    X_train = np.load(path+'train_samples_lpc.npz')
    y_train = np.load(path+'train_labels_lpc.npz')
    X_test = np.load(path+'test_samples_lpc.npz')
    y_test = np.load(path+'test_labels_lpc.npz')

    X_train, y_train, X_test, y_test = X_train['arr_0'], y_train['arr_0'], X_test['arr_0'], y_test['arr_0']

    # normalalize data and prep for Keras
    print('Processing images for Keras...')
    X_train, X_test = prep_train_test(X_train, X_test)

    # run CNN
    print('Fitting model...')
    random_forest(X_train, y_train, X_test, y_test)
