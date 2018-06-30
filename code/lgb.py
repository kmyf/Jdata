# -*- coding: utf-8 -*-
"""
@author: keming
"""
import pandas as pd
import numpy as np
from model import fit_predict_1, fit_predict_2
from datetime import datetime, timedelta
from fea_exact import gen_vali
import time

s_time = time.time()

a = str(datetime.now())
str_time = a[5:7] + '_' + a[8:10] + '_' + a[11:13] + '_' + a[14:16]

weight = []
for i in range(1, 50001):
    weight.append(1 / (1 + np.log(i)))


def score(df):
    sub = df.sort_values('o_num', ascending=False).reset_index(drop='index')
    sub['label_1'] = [1 if x > 0 else 0 for x in sub['label_1']]
    sub = sub.loc[:49999, :]
    sub['weight'] = weight

    s1 = sum(sub['label_1'] * sub['weight']) / 4674.323

    s2 = 0.0
    df = df[df['label_1'] > 0].reset_index(drop='index')
    for i in range(sub.shape[0]):
        if sub.loc[i, 'label_1'] > 0:
            s2 += 10.0 / ((sub.loc[i, 'label_2'] - np.round(sub.loc[i, 'pred_date'])) ** 2 + 10)

    s2 = s2 / df.shape[0]

    return s1, s2


def get_train():
    # 读取数据
    train = pd.read_csv(r'../input/vali_train_addnewfea_2.csv')
    test = pd.read_csv(r'../input/vali_test_addnewfea_2.csv')
    train['label_1'] = [1 if x > 0 else 0 for x in train['label_1']]
    test['label_1'] = [1 if x > 0 else 0 for x in test['label_1']]
    drop_column = ['user_id', 'label_1', 'label_2']
    result = test[['user_id']]  # 要提交的结果
    # 训练S1
    X = train.drop(drop_column, axis=1)
    X_pred = test.drop(drop_column, axis=1)
    y = train['label_1']
    result['o_num'], feat_imp_s1 = fit_predict_1(X, y, X_pred)
    feat_imp_s1.to_csv(r'../result/lgb_S1_%s.csv' % str_time, index=True, encoding='utf-8')
    # 训练S2
    y = train['label_2']
    result['pred_date'], feat_imp_s2 = fit_predict_2(X, y, X_pred)
    feat_imp_s2.to_csv(r'../result/lgb_S2_%s.csv' % str_time, index=True, encoding='utf-8')
    # 输出结果
    result[['label_1', 'label_2']] = test[['label_1', 'label_2']]
    s1, s2 = score(result)


    f = open(
        '../result/lgb_%s.txt' % str_time,
        'w')
    f.write('s1:' + str(s1) + 's2:' + str(s2) + 's:' + str(0.4 * s1 + 0.6 * s2))
    f.close()

    return feat_imp_s1, feat_imp_s2


def get_result():
    train = pd.read_csv(r'../input/test_train_addnewfea_2.csv')
    test = pd.read_csv(r'../input/test_test_addnewfea_2.csv')
    train['label_1'] = [1 if x > 0 else 0 for x in train['label_1']]
    test['label_1'] = [1 if x > 0 else 0 for x in test['label_1']]
    # one_hot_feature = ['sex','age']
    # train = pd.get_dummies(train, columns=one_hot_feature)
    # test = pd.get_dummies(test, columns=one_hot_feature)

    drop_column = ['user_id', 'label_1', 'label_2']

    sub = test[['user_id']]  # 要提交的结果

    # 训练S1
    X = train.drop(drop_column, axis=1)
    X_pred = test.drop(drop_column, axis=1)
    y = train['label_1']

    sub['o_num'], s1 = fit_predict_1(X, y, X_pred)

    X = train.drop(drop_column, axis=1)
    X_pred = test.drop(drop_column, axis=1)
    y = train['label_2']
    sub['pred_date'], s2 = fit_predict_2(X, y, X_pred)

    # 输出结果
    sub = sub.sort_values(by='o_num', ascending=False)[['user_id', 'pred_date']].reset_index(drop='index')
    sub['pred_date'] = sub['pred_date'].map(lambda day: datetime(2017, 9, 1) + timedelta(days=np.round(day - 1)))
    sub.loc[:49999, :].to_csv(r'../result/submission_%s.csv' % str_time, index=None, encoding='utf-8')

    return s1, s2


if __name__ == '__main__':
    feat_imp_s1, feat_imp_s2 = get_train()
    s1, s2 = get_result()
