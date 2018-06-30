# -*- coding: utf-8 -*-
"""
@author: keming
"""
import time

s_time = time.time()
import pandas as pd
from util import feat_nunique, feat_sum, feat_count, feat_max, load_data, feat_mean, feat_min, feat_median


order, action = load_data()
order['o_date'] = pd.to_datetime(order['o_date'])
action['a_date'] = pd.to_datetime(action['a_date'])


def create_feat(start_date, end_date, order, action, test=False):
    order = order.sort_values('o_date')
    action = action.sort_values('a_date')

    print(start_date)
    # 构建用户特征集
    df_label = pd.read_csv(r'../../../data/jdata_user_basic_info.csv')

    # 计算order和action与预测月份的时间差值
    enddate = pd.to_datetime(end_date)
    order.loc[:, 'day_gap'] = order['o_date'].apply(lambda x: (x - enddate).days).copy()
    action.loc[:, 'day_gap'] = action['a_date'].apply(lambda x: (x - enddate).days).copy()

    if test:
        df_label['label_1'] = -1
        df_label['label_2'] = -1
    else:
        # 预测目标月份的数据
        label_month = pd.to_datetime(end_date).month
        # ----------------------------------------------------------------------------------------------------------------------------------
        # 找到用户在目标月份最早的订单日期
        order_label = order[order['o_month'] == label_month]
        order_label = order_label[(order_label['cate'] == 30) | (order_label['cate'] == 101)]
        label = order_label.sort_values("o_date").drop_duplicates(["user_id"], keep="first")
        df_label = df_label.merge(label[['user_id', 'o_date']], on='user_id', how='left')

        df_label = feat_nunique(df_label, order_label, ['user_id'], 'o_id', 'label_1')
        df_label['label_1'] = [x if x <= 3 else 3 for x in df_label['label_1']]
        df_label['label_2'] = [x.day if pd.to_datetime(x) >= pd.to_datetime(start_date) else 0 for x in
                               df_label['o_date']]
        del df_label['o_date']

    # 用户总体特征
    order_tmp = order[order['day_gap'] < 0]
    action_tmp = action[action['day_gap'] < 0]

    df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'sku_id', 'sku_id_30_101_nunique')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                        'price', 'price_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                         'price', 'price_mean')
    df_label = feat_min(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                        'price', 'price_min')
    df_label = feat_max(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                        'price', 'price_max')
    df_label = feat_median(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                           'price', 'price_median')

    comment = order_tmp[order_tmp['score_level'] > 0]
    comment_1 = comment[comment['score_level'] == 1]
    comment_2 = comment[comment['score_level'] == 2]
    comment_3 = comment[comment['score_level'] == 3]
    df_label = feat_mean(df_label, comment[(comment['cate'] == 30) | (comment['cate'] == 101)], ['user_id'],
                         'score_level', 'user_comment_score_mean')
    df_label = feat_sum(df_label, comment_1[(comment_1['cate'] == 30) | (comment_1['cate'] == 101)], ['user_id'],
                        'score_level', 'user_comment_score_sum1')
    df_label = feat_sum(df_label, comment_2[(comment_2['cate'] == 30) | (comment_2['cate'] == 101)], ['user_id'],
                        'score_level', 'user_comment_score_sum2')
    df_label = feat_sum(df_label, comment_3[(comment_3['cate'] == 30) | (comment_3['cate'] == 101)], ['user_id'],
                        'score_level', 'user_comment_score_sum3')

    df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'o_id', 'o_id_30_101_nunique')
    df_label = feat_count(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                          'sku_id', 'o_sku_id_30_101_count')
    df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                        'o_sku_num', 'o_sku_num_30_101_sum')
    df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                         'o_day', 'day_30_101_mean')
    df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'o_date', 'o_date_30_101_nunique')
    df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'o_month', 'o_month_30_101_nunique')
    df_label = feat_count(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)], ['user_id'],
                          'sku_id', 'a_sku_id_30_101_count')
    df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)], ['user_id'],
                            'a_date', 'a_date_30_101_nunique')
    df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 1)], ['user_id'], 'a_date', 'a_date_1_nunique')
    df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 71)], ['user_id'], 'a_date',
                            'a_date_71_nunique')
    df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 83)], ['user_id'], 'a_date',
                            'a_date_46_nunique')
    df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 46)], ['user_id'], 'a_date',
                            'a_date_83_nunique')
    df_label = feat_mean(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)], ['user_id'],
                         'a_day', 'a_day_30_101_mean')
    df_label = feat_sum(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)], ['user_id'],
                        'a_num', 'a_num_30_101_sum')
    df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)], ['user_id'],
                            'a_month', 'a_month_30_101_nunique')

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
    # 时间窗来提取特征
    for i in [7, 14, 30, 90, 150]:
        print(i)
        order_tmp = order[(order['day_gap'] >= -i) & (order['day_gap'] < 0)]
        action_tmp = action[(action['day_gap'] >= -i) & (action['day_gap'] < 0)]

        a = "AD" + str(i) + "_"
        o = "OD" + str(i) + "_"

        # 这是有用的特征
        df_label = feat_count(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                              'para_3_-1', o + '30_101_para_3_-1_count')


        # price特征
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'price', o + 'price_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                             'price', o + 'price_mean')
        df_label = feat_min(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'price', o + 'price_min')
        df_label = feat_max(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'price', o + 'price_max')
        df_label = feat_median(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                               'price', o + 'price_median')

        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                             'para_1', o + 'para_1_30_101_mean')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'para_1', o + 'para_1_30_101_sum')
        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30)], ['user_id'],
                             'para_1', o + 'para_1_30_mean')

        # para_1特征
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

       # 订单特征
        df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)],
                                ['user_id'], 'o_id', o + 'o_id_30_101_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 30], ['user_id'], 'o_id',
                                o + 'o_id_30_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 101], ['user_id'], 'o_id',
                                o + 'o_id_101_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 1], ['user_id'], 'o_id', o + 'o_id_1_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 71], ['user_id'], 'o_id',
                                o + 'o_id_71_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 46], ['user_id'], 'o_id',
                                o + 'o_id_46_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 83], ['user_id'], 'o_id',
                                o + 'o_id_83_nunique')
        df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] != 30) & (order_tmp['cate'] != 101)],
                                ['user_id'], 'o_id', o + 'o_id_other_nunique')

        # 商品特征
        df_label = feat_count(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                              'sku_id', o + 'sku_id_30_101_count')
        df_label = feat_count(df_label, order_tmp[order_tmp['cate'] == 30], ['user_id'], 'sku_id',
                              o + 'sku_id_30_count')
        df_label = feat_count(df_label, order_tmp[order_tmp['cate'] == 101], ['user_id'], 'sku_id',
                              o + 'sku_id_101_count')
        df_label = feat_count(df_label, order_tmp[order_tmp['cate'] == 1], ['user_id'], 'sku_id', o + 'sku_id_1_count')
        df_label = feat_count(df_label, order_tmp[order_tmp['cate'] == 71], ['user_id'], 'sku_id',
                              o + 'sku_id_71_count')
        df_label = feat_count(df_label, order_tmp[order_tmp['cate'] == 46], ['user_id'], 'sku_id',
                              o + 'sku_id_46_count')
        df_label = feat_count(df_label, order_tmp[order_tmp['cate'] == 83], ['user_id'], 'sku_id',
                              o + 'sku_id_83_count')
        df_label = feat_count(df_label, order_tmp[(order_tmp['cate'] != 30) & (order_tmp['cate'] != 101)], ['user_id'],
                              'sku_id', o + 'sku_id_other_count')


        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'o_sku_num', o + 'o_sku_num_30_101_count')
        df_label = feat_sum(df_label, order_tmp[order_tmp['cate'] == 30], ['user_id'], 'o_sku_num',
                            o + 'o_sku_num_30_count')
        df_label = feat_sum(df_label, order_tmp[order_tmp['cate'] == 101], ['user_id'], 'o_sku_num',
                            o + 'o_sku_num_101_count')
        df_label = feat_sum(df_label, order_tmp[order_tmp['cate'] == 1], ['user_id'], 'o_sku_num',
                            o + 'o_sku_num_1_count')
        df_label = feat_sum(df_label, order_tmp[order_tmp['cate'] == 71], ['user_id'], 'o_sku_num',
                            o + 'o_sku_num_71_count')
        df_label = feat_sum(df_label, order_tmp[order_tmp['cate'] == 46], ['user_id'], 'o_sku_num',
                            o + 'o_sku_num_46_count')
        df_label = feat_sum(df_label, order_tmp[order_tmp['cate'] == 83], ['user_id'], 'o_sku_num',
                            o + 'o_sku_num_83_count')
        df_label = feat_sum(df_label, order_tmp[(order_tmp['cate'] != 30) & (order_tmp['cate'] != 101)], ['user_id'],
                            'o_sku_num', o + 'o_sku_num_other_count')


        df_label[o + 'o_date_30_101_firstday'] = \
        order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)].drop_duplicates(['user_id'], keep='first')[
            'o_day']

        df_label = feat_max(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                            'o_day', o + 'day_30_101_max')
        df_label = feat_max(df_label, order_tmp[order_tmp['cate'] == 30], ['user_id'], 'o_day', o + 'day_30_max')
        df_label = feat_max(df_label, order_tmp[order_tmp['cate'] == 101], ['user_id'], 'o_day', o + 'day_101_max')


        df_label = feat_mean(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)], ['user_id'],
                             'o_day', o + 'day_30_101_mean')
        df_label = feat_mean(df_label, order_tmp[order_tmp['cate'] == 30], ['user_id'], 'o_day', o + 'day_30_mean')
        df_label = feat_mean(df_label, order_tmp[order_tmp['cate'] == 101], ['user_id'], 'o_day', o + 'day_101_mean')


        df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)],
                                ['user_id'], 'o_date', o + 'o_date_30_101_mean')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 30], ['user_id'], 'o_date',
                                o + 'o_date_30_mean')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 101], ['user_id'], 'o_date',
                                o + 'o_date_101_mean')

        df_label = feat_nunique(df_label, order_tmp[(order_tmp['cate'] == 30) | (order_tmp['cate'] == 101)],
                                ['user_id'], 'o_month', o + 'o_month_30_101_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 30], ['user_id'], 'o_month',
                                o + 'o_month_30_nunique')
        df_label = feat_nunique(df_label, order_tmp[order_tmp['cate'] == 101], ['user_id'], 'o_month',
                                o + 'o_month_101_nunique')

        # 用户点击收藏行为
        action_tmp_type_1 = action_tmp[action_tmp['a_type'] == 1]
        action_tmp_type_2 = action_tmp[action_tmp['a_type'] == 2]


        df_label = feat_count(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)],
                              ['user_id'], 'sku_id', a + 'sku_id_30_101_count')
        df_label = feat_count(df_label, action_tmp[action_tmp['cate'] == 30], ['user_id'], 'sku_id',
                              a + 'sku_id_30_count')
        df_label = feat_count(df_label, action_tmp[action_tmp['cate'] == 101], ['user_id'], 'sku_id',
                              a + 'sku_id_101_count')
        df_label = feat_count(df_label, action_tmp[action_tmp['cate'] == 1], ['user_id'], 'sku_id',
                              a + 'sku_id_1_count')
        df_label = feat_count(df_label, action_tmp[action_tmp['cate'] == 71], ['user_id'], 'sku_id',
                              a + 'sku_id_71_count')
        df_label = feat_count(df_label, action_tmp[action_tmp['cate'] == 46], ['user_id'], 'sku_id',
                              a + 'sku_id_46_count')
        df_label = feat_count(df_label, action_tmp[action_tmp['cate'] == 83], ['user_id'], 'sku_id',
                              a + 'sku_id_83_count')

        # 用户当月行为总次数
        df_label = feat_sum(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)], ['user_id'],
                            'a_num', a + 'a_num_30_101_sum')
        df_label = feat_sum(df_label, action_tmp[action_tmp['cate'] == 30], ['user_id'], 'a_num', a + 'a_num_30_sum')
        df_label = feat_sum(df_label, action_tmp[action_tmp['cate'] == 101], ['user_id'], 'a_num', a + 'a_num_101_sum')
        df_label = feat_sum(df_label, action_tmp[action_tmp['cate'] == 1], ['user_id'], 'a_num', a + 'a_num_1_sum')
        df_label = feat_sum(df_label, action_tmp[action_tmp['cate'] == 71], ['user_id'], 'a_num', a + 'a_num_71_sum')
        df_label = feat_sum(df_label, action_tmp[action_tmp['cate'] == 46], ['user_id'], 'a_num', a + 'a_num_46_sum')
        df_label = feat_sum(df_label, action_tmp[action_tmp['cate'] == 83], ['user_id'], 'a_num', a + 'a_num_83_sum')
        df_label = feat_sum(df_label, action_tmp[(action_tmp['cate'] != 30) & (action_tmp['cate'] != 101)], ['user_id'],
                            'a_num', o + 'a_num_other_count')

        # 用户当月行为1次数
        df_label = feat_sum(df_label,
                            action_tmp_type_1[(action_tmp_type_1['cate'] == 30) | (action_tmp_type_1['cate'] == 101)],
                            ['user_id'], 'a_num', a + 'a_num_30_101_type_1_sum')
        df_label = feat_sum(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 30], ['user_id'], 'a_num',
                            a + 'a_num_30_type_1_sum')
        df_label = feat_sum(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 101], ['user_id'], 'a_num',
                            a + 'a_num_101_type_1_sum')
        df_label = feat_sum(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 1], ['user_id'], 'a_num',
                            a + 'a_num_1_type_1_sum')
        df_label = feat_sum(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 71], ['user_id'], 'a_num',
                            a + 'a_num_71_type_1_sum')
        df_label = feat_sum(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 46], ['user_id'], 'a_num',
                            a + 'a_num_46_type_1_sum')
        df_label = feat_sum(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 83], ['user_id'], 'a_num',
                            a + 'a_num_83_type_1_sum')
        df_label = feat_sum(df_label,
                            action_tmp_type_1[(action_tmp_type_1['cate'] != 30) & (action_tmp_type_1['cate'] != 101)],
                            ['user_id'], 'a_num', a + 'a_num_other_type_1_count')

        # 用户当月行为2次数
        df_label = feat_sum(df_label,
                            action_tmp_type_2[(action_tmp_type_2['cate'] == 30) | (action_tmp_type_2['cate'] == 101)],
                            ['user_id'], 'a_num', a + 'a_num_30_101_type_2_sum')
        df_label = feat_sum(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 30], ['user_id'], 'a_num',
                            a + 'a_num_30_type_2_sum')
        df_label = feat_sum(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 101], ['user_id'], 'a_num',
                            a + 'a_num_101_type_2_sum')
        df_label = feat_sum(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 1], ['user_id'], 'a_num',
                            a + 'a_num_1_type_2_sum')
        df_label = feat_sum(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 71], ['user_id'], 'a_num',
                            a + 'a_num_71_type_2_sum')
        df_label = feat_sum(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 46], ['user_id'], 'a_num',
                            a + 'a_num_46_type_2_sum')
        df_label = feat_sum(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 83], ['user_id'], 'a_num',
                            a + 'a_num_83_type_2_sum')
        df_label = feat_sum(df_label,
                            action_tmp_type_2[(action_tmp_type_2['cate'] != 30) & (action_tmp_type_2['cate'] != 101)],
                            ['user_id'], 'a_num', a + 'a_num_other_type_2_count')
        ####################################################################################################################################
        # 用户当月浏览行为次数
        df_label = feat_count(df_label,
                              action_tmp_type_1[(action_tmp_type_1['cate'] == 30) | (action_tmp_type_1['cate'] == 101)],
                              ['user_id'], 'sku_id', a + 'sku_id_type1_30_101_count')
        df_label = feat_count(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 30], ['user_id'], 'sku_id',
                              a + 'sku_id_type1_30_count')
        df_label = feat_count(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 101], ['user_id'], 'sku_id',
                              a + 'sku_id_type1_101_count')
        df_label = feat_count(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 1], ['user_id'], 'sku_id',
                              a + 'sku_id_type1_1_count')
        df_label = feat_count(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 71], ['user_id'], 'sku_id',
                              a + 'sku_id_type1_71_count')
        df_label = feat_count(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 46], ['user_id'], 'sku_id',
                              a + 'sku_id_type1_46_count')
        df_label = feat_count(df_label, action_tmp_type_1[action_tmp_type_1['cate'] == 83], ['user_id'], 'sku_id',
                              a + 'sku_id_type1_83_count')

        ####################################################################################################################################
        # 用户当月收藏行为次数
        df_label = feat_count(df_label,
                              action_tmp_type_2[(action_tmp_type_2['cate'] == 30) | (action_tmp_type_2['cate'] == 101)],
                              ['user_id'], 'sku_id', a + 'sku_id_type2_30_101_count')
        df_label = feat_count(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 30], ['user_id'], 'sku_id',
                              a + 'sku_id_type2_30_count')
        df_label = feat_count(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 101], ['user_id'], 'sku_id',
                              a + 'sku_id_type2_101_count')
        df_label = feat_count(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 1], ['user_id'], 'sku_id',
                              a + 'sku_id_type2_1_count')
        df_label = feat_count(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 71], ['user_id'], 'sku_id',
                              a + 'sku_id_type2_71_count')
        df_label = feat_count(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 46], ['user_id'], 'sku_id',
                              a + 'sku_id_type2_46_count')
        df_label = feat_count(df_label, action_tmp_type_2[action_tmp_type_2['cate'] == 83], ['user_id'], 'sku_id',
                              a + 'sku_id_type2_83_count')

        ####################################################################################################################################
        # 用户当月行为天数
        df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)],
                                ['user_id'], 'a_date', a + 'a_date_30_101_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 30], ['user_id'], 'a_date',
                                a + 'a_date_30_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 101], ['user_id'], 'a_date',
                                a + 'a_date_101_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 1], ['user_id'], 'a_date',
                                a + 'a_date_1_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 71], ['user_id'], 'a_date',
                                a + 'a_date_71_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 46], ['user_id'], 'a_date',
                                a + 'a_date_146_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 83], ['user_id'], 'a_date',
                                a + 'a_date_83_nunique')

        ####################################################################################################################################
        # 用户当月行为商品数
        df_label = feat_nunique(df_label, action_tmp[(action_tmp['cate'] == 30) | (action_tmp['cate'] == 101)],
                                ['user_id'], 'sku_id', a + 'sku_id_30_101_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 30], ['user_id'], 'sku_id',
                                a + 'sku_id_30_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 101], ['user_id'], 'sku_id',
                                a + 'sku_id_101_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 1], ['user_id'], 'sku_id',
                                a + 'sku_id_1_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 71], ['user_id'], 'sku_id',
                                a + 'sku_id_71_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 46], ['user_id'], 'sku_id',
                                a + 'sku_id_46_nunique')
        df_label = feat_nunique(df_label, action_tmp[action_tmp['cate'] == 83], ['user_id'], 'sku_id',
                                a + 'sku_id_83_nunique')

    print(df_label.shape)
    return df_label


def gen_vali():
    train = create_feat('2017-5-1', '2017-6-1', order, action)
    train = pd.concat([train, create_feat('2017-4-1', '2017-5-1', order, action)])
    train = pd.concat([train, create_feat('2017-3-1', '2017-4-1', order, action)])
    train = pd.concat([train, create_feat('2017-2-1', '2017-3-1', order, action)])
    train = pd.concat([train, create_feat('2016-1-1', '2016-2-1', order, action)])
    # 验证训练集
    test = create_feat('2017-6-1', '2017-7-1', order, action)
    train = pd.concat([train, test])
    train.to_csv(r'../input/vali_train.csv', index=None)
    # 验证测试集
    test = create_feat('2017-7-1', '2017-8-1', order, action)
    test.to_csv(r'../input/vali_test.csv', index=None)

    # 测试训练集
    train = pd.concat([train, test])
    train.to_csv(r'../input/test_train.csv', index=None)
    # 测试测试集
    test = create_feat('2017-8-1', '2017-9-1', order, action)
    test.to_csv(r'../input/test_test.csv', index=None)

if __name__ == '__main__':
    gen_vali()