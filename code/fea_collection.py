# -*- coding: utf-8 -*-
import time

s_time = time.time()
import pandas as pd
from util import feat_nunique, feat_sum, feat_count, feat_max, load_data, feat_mean, feat_min, feat_median

order, action = load_data()
order['o_date'] = pd.to_datetime(order['o_date'])
action['a_date'] = pd.to_datetime(action['a_date'])
order = order.sort_values('o_date')
action = action.sort_values('a_date')


def create_feat(start_date, end_date, order, action, test=False):
    print(start_date)
    df_label = pd.read_csv(r'../../../data/jdata_user_basic_info.csv')
    drop_label = ['user_id', 'age', 'sex', 'user_lv_cd']

    enddate = pd.to_datetime(end_date)
    order.loc[:, 'day_gap'] = order['o_date'].apply(lambda x: (x - enddate).days).copy()
    action.loc[:, 'day_gap'] = action['a_date'].apply(lambda x: (x - enddate).days).copy()

    order_tmp = order[order['day_gap'] < 0]
    action_tmp = action[action['day_gap'] < 0]

    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                         'para_1', 'para_1_30_101_mean')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                        'para_1', 'para_1_30_101_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30)], ['user_id'],
                         'para_1', 'para_1_30_mean')
    # new
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30)], ['user_id'],
                        'para_1', 'para_1_30_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 101)], ['user_id'],
                         'para_1', 'para_1_101_mean')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 101)], ['user_id'],
                        'para_1', 'para_1_101_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 1)], ['user_id'],
                         'para_1', 'para_1_1_mean')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 1)], ['user_id'],
                        'para_1', 'para_1_1_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 71)], ['user_id'],
                         'para_1', 'para_1_71_mean')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 71)], ['user_id'],
                        'para_1', 'para_1_71_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 83)], ['user_id'],
                         'para_1', 'para_1_83_mean')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 83)], ['user_id'],
                        'para_1', 'para_1_83_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 64)], ['user_id'],
                         'para_1', 'para_1_64_mean')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 64)], ['user_id'],
                        'para_1', 'para_1_64_sum')
    df_label = feat_mean(df_label, order_tmp, ['user_id'],
                         'para_1', 'para_1_mean')
    # new_2
    for i in [7, 14, 30, 90, 150]:
        print(i)
        order_tmp = order[(order['day_gap'] >= -i) & (order['day_gap'] < 0)]
        action_tmp = action[(action['day_gap'] >= -i) & (action['day_gap'] < 0)]
        a = "AD" + str(i) + "_"
        o = "OD" + str(i) + "_"
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                             'para_1', o + 'para_1_30_101_mean')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'para_1', o + 'para_1_30_101_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30)], ['user_id'],
                             'para_1', o + 'para_1_30_mean')

        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30)], ['user_id'],
                            'para_1', o + 'para_1_30_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 101)], ['user_id'],
                             'para_1', o + 'para_1_101_mean')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 101)], ['user_id'],
                            'para_1', o + 'para_1_101_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 1)], ['user_id'],
                             'para_1', o + 'para_1_1_mean')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 1)], ['user_id'],
                            'para_1', o + 'para_1_1_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 71)], ['user_id'],
                             'para_1', o + 'para_1_71_mean')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 71)], ['user_id'],
                            'para_1', o + 'para_1_71_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 83)], ['user_id'],
                             'para_1', o + 'para_1_83_mean')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 83)], ['user_id'],
                            'para_1', o + 'para_1_83_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 64)], ['user_id'],
                             'para_1', o + 'para_1_64_mean')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 64)], ['user_id'],
                            'para_1', o + 'para_1_64_sum')
        df_label = feat_mean(df_label, order_tmp, ['user_id'],
                             'para_1', o + 'para_1_mean')

    '''
    df_label = feat_mean(df_label,action_tmp[(action_tmp['cate']==30)|(action_tmp['cate']==101)],['user_id'],'a_day','a_day_30_101_mean')
    df_label = feat_sum(df_label,action_tmp[(action_tmp['cate']==30)|(action_tmp['cate']==101)],['user_id'],'a_num','a_num_30_101_sum')
    df_label = feat_nunique(df_label,action_tmp[(action_tmp['cate']==30)|(action_tmp['cate']==101)],['user_id'],'a_month','a_month_30_101_nunique')
    for i in [7,14,30,90,150]:
        print (i)
        order_tmp = order[(order['day_gap']>=-i)&(order['day_gap']<0)]
        action_tmp = action[(action['day_gap']>=-i)&(action['day_gap']<0)]

        a = "AD"+str(i)+"_"
        o = "OD"+str(i)+"_"
        #zhegeshiyouyongde
        df_label=feat_count(df_label,order_tmp[(order_tmp['cate']==30)|(order_tmp['cate']==101)],['user_id'],'para_3_-1',o+'30_101_para_3_-1_count')



        action_tmp_type_1 = action_tmp[action_tmp['a_type']==1]
        action_tmp_type_2 = action_tmp[action_tmp['a_type']==2]

        #用户当月(30,101),30,101行为总次数
        df_label = feat_sum(df_label,action_tmp[(action_tmp['cate']==30)|(action_tmp['cate']==101)],['user_id'],'a_num',a+'a_num_30_101_sum')
        df_label = feat_sum(df_label,action_tmp[action_tmp['cate']==30],['user_id'],'a_num',a+'a_num_30_sum')
        df_label = feat_sum(df_label,action_tmp[action_tmp['cate']==101],['user_id'],'a_num',a+'a_num_101_sum')
        df_label = feat_sum(df_label,action_tmp[action_tmp['cate']==1],['user_id'],'a_num',a+'a_num_1_sum')
        df_label = feat_sum(df_label,action_tmp[action_tmp['cate']==71],['user_id'],'a_num',a+'a_num_71_sum')
        df_label = feat_sum(df_label,action_tmp[action_tmp['cate']==46],['user_id'],'a_num',a+'a_num_46_sum')
        df_label = feat_sum(df_label,action_tmp[action_tmp['cate']==83],['user_id'],'a_num',a+'a_num_83_sum')
        df_label = feat_sum(df_label,action_tmp[(action_tmp['cate']!=30)&(action_tmp['cate']!=101)],['user_id'],'a_num',o+'a_num_other_count')

        #用户当月(30,101),30,101行为1次数
        df_label = feat_sum(df_label,action_tmp_type_1[(action_tmp_type_1['cate']==30)|(action_tmp_type_1['cate']==101)],['user_id'],'a_num',a+'a_num_30_101_type_1_sum')
        df_label = feat_sum(df_label,action_tmp_type_1[action_tmp_type_1['cate']==30],['user_id'],'a_num',a+'a_num_30_type_1_sum')
        df_label = feat_sum(df_label,action_tmp_type_1[action_tmp_type_1['cate']==101],['user_id'],'a_num',a+'a_num_101_type_1_sum')
        df_label = feat_sum(df_label,action_tmp_type_1[action_tmp_type_1['cate']==1],['user_id'],'a_num',a+'a_num_1_type_1_sum')
        df_label = feat_sum(df_label,action_tmp_type_1[action_tmp_type_1['cate']==71],['user_id'],'a_num',a+'a_num_71_type_1_sum')
        df_label = feat_sum(df_label,action_tmp_type_1[action_tmp_type_1['cate']==46],['user_id'],'a_num',a+'a_num_46_type_1_sum')
        df_label = feat_sum(df_label,action_tmp_type_1[action_tmp_type_1['cate']==83],['user_id'],'a_num',a+'a_num_83_type_1_sum')
        df_label = feat_sum(df_label,action_tmp_type_1[(action_tmp_type_1['cate']!=30)&(action_tmp_type_1['cate']!=101)],['user_id'],'a_num',a+'a_num_other_type_1_count')

        #用户当月(30,101),30,101行为2次数
        df_label = feat_sum(df_label,action_tmp_type_2[(action_tmp_type_2['cate']==30)|(action_tmp_type_2['cate']==101)],['user_id'],'a_num',a+'a_num_30_101_type_2_sum')
        df_label = feat_sum(df_label,action_tmp_type_2[action_tmp_type_2['cate']==30],['user_id'],'a_num',a+'a_num_30_type_2_sum')
        df_label = feat_sum(df_label,action_tmp_type_2[action_tmp_type_2['cate']==101],['user_id'],'a_num',a+'a_num_101_type_2_sum')
        df_label = feat_sum(df_label,action_tmp_type_2[action_tmp_type_2['cate']==1],['user_id'],'a_num',a+'a_num_1_type_2_sum')
        df_label = feat_sum(df_label,action_tmp_type_2[action_tmp_type_2['cate']==71],['user_id'],'a_num',a+'a_num_71_type_2_sum')
        df_label = feat_sum(df_label,action_tmp_type_2[action_tmp_type_2['cate']==46],['user_id'],'a_num',a+'a_num_46_type_2_sum')
        df_label = feat_sum(df_label,action_tmp_type_2[action_tmp_type_2['cate']==83],['user_id'],'a_num',a+'a_num_83_type_2_sum')
        df_label = feat_sum(df_label,action_tmp_type_2[(action_tmp_type_2['cate']!=30)&(action_tmp_type_2['cate']!=101)],['user_id'],'a_num',a+'a_num_other_type_2_count')
    '1''

        '2''
        # new2 meiyongba

        df_label = feat_sum(df_label,order_tmp[(order_tmp['cate']==30)|(order_tmp['cate']==101)],['user_id'],'o_sku_num',o+'o_sku_num_30_101_count')
        df_label = feat_sum(df_label,order_tmp[order_tmp['cate']==30],['user_id'],'o_sku_num',o+'o_sku_num_30_count')
        df_label = feat_sum(df_label,order_tmp[order_tmp['cate']==101],['user_id'],'o_sku_num',o+'o_sku_num_101_count')
        df_label = feat_sum(df_label,order_tmp[order_tmp['cate']==1],['user_id'],'o_sku_num',o+'o_sku_num_1_count')
        df_label = feat_sum(df_label,order_tmp[order_tmp['cate']==71],['user_id'],'o_sku_num',o+'o_sku_num_71_count')
        df_label = feat_sum(df_label,order_tmp[order_tmp['cate']==46],['user_id'],'o_sku_num',o+'o_sku_num_46_count')
        df_label = feat_sum(df_label,order_tmp[order_tmp['cate']==83],['user_id'],'o_sku_num',o+'o_sku_num_83_count')
        df_label = feat_sum(df_label,order_tmp[(order_tmp['cate']!=30)&(order_tmp['cate']!=101)],['user_id'],'o_sku_num',o+'o_sku_num_other_count')

        df_label[o+'order_persent_count_30_101_type1'] = df_label[o+'o_sku_num_30_101_count']/df_label[a+'a_num_30_101_type_2_sum']
        df_label[o+'order_persent_count_30_101_type2'] = df_label[o+'o_sku_num_30_101_count']/df_label[a+'a_num_30_101_type_2_sum']
        df_label[o+'order_persent_count_30_type1'] = df_label[o+'o_sku_num_30_count']/df_label[a+'a_num_30_type_1_sum']
        df_label[o+'order_persent_count_30_type2'] = df_label[o+'o_sku_num_30_count']/df_label[a+'a_num_30_type_2_sum']
        df_label[o+'order_persent_count_101_type1'] = df_label[o+'o_sku_num_101_count']/df_label[a+'a_num_101_type_1_sum']
        df_label[o+'order_persent_count_101_type2'] = df_label[o+'o_sku_num_101_count']/df_label[a+'a_num_101_type_2_sum']
        df_label[o+'order_persent_count_1_type1'] = df_label[o+'o_sku_num_1_count']/df_label[a+'a_num_1_type_1_sum']
        df_label[o+'order_persent_count_1_type2'] = df_label[o+'o_sku_num_1_count']/df_label[a+'a_num_1_type_2_sum']
        df_label[o+'order_persent_count_71_type1'] = df_label[o+'o_sku_num_71_count']/df_label[a+'a_num_71_type_1_sum']
        df_label[o+'order_persent_count_71_type2'] = df_label[o+'o_sku_num_71_count']/df_label[a+'a_num_71_type_2_sum']
        df_label[o+'order_persent_count_46_type1'] = df_label[o+'o_sku_num_46_count']/df_label[a+'a_num_46_type_1_sum']
        df_label[o+'order_persent_count_46_type2'] = df_label[o+'o_sku_num_46_count']/df_label[a+'a_num_46_type_2_sum']
        df_label[o+'order_persent_count_83_type1'] = df_label[o+'o_sku_num_83_count']/df_label[a+'a_num_83_type_1_sum']
        df_label[o+'order_persent_count_83_type2'] = df_label[o+'o_sku_num_83_count']/df_label[a+'a_num_83_type_2_sum']
        df_label[o+'order_persent_count_other_type1'] = df_label[o+'o_sku_num_other_count']/df_label[a+'a_num_other_type_1_count']
        df_label[o+'order_persent_count_other_type2'] = df_label[o+'o_sku_num_other_count']/df_label[a+'a_num_other_type_2_count']

        drop_label_2 = [o+'o_sku_num_30_101_count',o+'o_sku_num_30_count',o+'o_sku_num_101_count',o+'o_sku_num_1_count',o+'o_sku_num_71_count',o+'o_sku_num_46_count',o+'o_sku_num_83_count',o+'o_sku_num_other_count']
        df_label = df_label.drop(drop_label_2,axis=1)
        '''

    return_label = df_label.drop(drop_label, axis=1)
    print(return_label, 11111111111111111)
    return return_label


def gen_vali():
    train = create_feat('2017-5-1', '2017-6-1', order, action)
    train = pd.concat([train, create_feat('2017-4-1', '2017-5-1', order, action)])
    train = pd.concat([train, create_feat('2017-3-1', '2017-4-1', order, action)])
    train = pd.concat([train, create_feat('2017-2-1', '2017-3-1', order, action)])
    train = pd.concat([train, create_feat('2016-1-1', '2016-2-1', order, action)])

    test = create_feat('2017-6-1', '2017-7-1', order, action)

    train = pd.concat([train, test])
    # 旧的训练集加到新的训练集上
    old_train = pd.read_csv('../input/vali_train_addnewfea.csv')
    old_train['aaaaa'] = [i for i in range(len(old_train))]
    train['aaaaa'] = [i for i in range(len(old_train))]
    train = pd.merge(old_train, train, how='left').drop(['aaaaa'], axis=1)
    train.to_csv(r'../input/vali_train_addnewfea_2.csv', index=None)
    print('train.shappe', train.shape)

    test = create_feat('2017-7-1', '2017-8-1', order, action)
    # 旧的测试集加到新的测试集上
    old_test = pd.read_csv('../input/vali_test_addnewfea.csv')
    old_test['aaaaa'] = [i for i in range(len(old_test))]
    test['aaaaa'] = [i for i in range(len(old_test))]
    test = pd.merge(old_test, test, how='left').drop(['aaaaa'], axis=1)
    print('test.shape', test.shape)
    test.to_csv(r'../input/vali_test_addnewfea_2.csv', index=None)

    train = pd.concat([train, test])
    train.to_csv(r'../input/test_train_addnewfea_2.csv', index=None)
    # 5月做预测

    test = create_feat('2017-8-1', '2017-9-1', order, action)
    # 旧的测试集加到新的测试集上
    old_test = pd.read_csv('../input/test_test_addnewfea.csv')
    old_test['aaaaa'] = [i for i in range(len(old_test))]
    test['aaaaa'] = [i for i in range(len(old_test))]
    test = pd.merge(old_test, test, how='left').drop(['aaaaa'], axis=1)
    print('test.shape', test.shape)
    test.to_csv(r'../input/test_test_addnewfea_2.csv', index=None)


if __name__ == '__main__':
    gen_vali()

    print(time.time() - s_time)