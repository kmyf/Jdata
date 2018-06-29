# Jdata
如期而至-用户购买时间预测
# 最终成绩
0.3438(S1:0.4802/S2:0.2529) \
排名26，参赛人数5182
# 队伍
队伍名称 MADE \
队长：kmyf  队员：chenxj
# 说明
1. 运行环境
python3
2. 需要安装的包
lightgbm、pandas、numpy、sklearn
3. 运行
将数据集放在data文件夹下，运行lgb.py

# 前言
这是第一次取得还算自己满意的成绩（虽然在大佬看来不是很好），感觉这个比赛也付出了很多，花了很多时间，从找特征找模型，尝试过很多方法，也参考了很多大佬的开源，所以在这里也将自己的代码开源出来，向开源致敬

# 赛题回顾
## 竞赛概述 
京东多年来在保持高速发展的同时，沉淀了数亿的忠实用户，积累了海量的真实数据。如何从历史数据中找出规律，高效解决客户实际问题、提升客户购物体验，是大数据应用在精准营销中的关键问题，也是所有电商平台在做智能化升级时所需要的核心技术。未来，京东将更多的应用先进算法技术，秉承开放、赋能战略构想，为无界零售提供零售基础设施。
本次大赛京东携手中国信息通信研究院，中关村加一战略新兴产业人才发展中心，以“创响中国”在大数据领域落地为契机，旨在激发算法创新，释放数据价值，培养数据人才。希望各界算法精英能够利用脱敏后的京东真实用户历史行为数据，自建算法模型，预测热销品类的用户购买时间，搏击浪潮之巅，勇夺桂冠，赢得高额奖金。

## 赛题链接 
https://jdata.jd.com/html/detail.html?id=2

# 基本解决方案
本题要预测两个部分，s1为购买的用户，s2为用户购买的时间
## 解题思路
这个题目首先要预测的是购买的用户，只有s1比较高的情况下，s2自然而然的比较高。因此我们觉得这题可以从s1入手，先预测出购买的用户，再来预测s2用户购买的时间。
## 数据预览
拿A榜数据为例，下同
1. sku包含sku_id,price,cate,para_1,para_2,para_3 \
其中sku_id有99412个，cate有六种，我们预测的为两种，因此提取特征时按照不同的cate来提取特征是有帮助的。para_1类别很多，para_2、para_3类别只有几种，并且也都去敏感了，提取特征的时候将其独热编码然后来判断特征是否重要。
2. action包含user_id,sku_id,a_date,a_num,a_type \
其中action有6944141数据，用到了99133个sku_id，有365天的数据，93453个用户ID，两种用户类型。
3. basic_info包含user_id,age,sex,user_lv_cd
其中有98924个用户ID，年龄和性别都可以独热编码来作为特征
4.comment_score包含user_id,comment_create_tm,o_id,score_level
其中有224284条评论，用到了42191个user_id,222972种comment_create_tm，191462个o_id
5. order包含了user_id,sku_id,o_id,o_date,o_area,o_sku_num
其中660933种o_id，98924种user_id，25474种sku_id，365天的数据，31个o_area，51种o_sku_num 
A榜的数据没有缺失值，B榜的数据有，拿到数据后先看下数据的分布，数据自带的特征，并且这些数据包含了节假日，6.18和11.11这两天的订单量远大于其他天数。

## 特征寻找
寻找特征想到了两种思路，一种是根据用户来寻找特征（另一种是根据行为来寻找特征，在看其他开源的时候有见到过）\
提取特征分为两类，一类根据用户的订单来提取特征，一类根据用户的行为来提取特征\
特征提取方法
1. 时间窗口法，提取预测时间前一段时间内的特征，如用户近半个月的行为，一个月的行为，三个月的行为等，来预测下个月的行为
2. 概率累计法，通过用户的统计特征来计算下单的概率，如用户的点击/收藏/下单行为转化成下单的概率
3. 时间间隔法，用户本次的行为/下单对应着下一次下单的时间间隔

## 模型选择
使用了lgb和xgb模型，但是由于xgb模型在线下表现的没有lgb好，所以最终使用的是lgb模型。当然也尝试过其他模型，比如ffm,deepfm，由于使用使用不熟练并且模型的一些限制所以效果不是很好。 \
模型选择上面我们花的时间还是比较少的，使用lgb模型基本没怎么调参，大部分时间还是花在了找特征上面。寻找特征的时候使用单模型，根据单模型的结果来判断特征是否有用。在提交的时候使用了stacking和bagging模型融合方法，线下训练能在原来的基础上s1增加个两个千分点。\
lgb调参只尝试了修改学习率，修改二分类和回归问题，修改了对应的损失函数。
# 后记
比赛结束后看别人思路发现把s2的目标函数改一下，不用mse，使用s2的评分函数，可以让s2提高3个百分点，这一点当时没有考虑到，但确实是可行的。
s2我们确实没怎么好好考虑，训练时也是一直在关注着s1，所以s2的分数随着s1的增加而增加。理论上s2与s1分开训练，使用不同的特征集会比较好，我们还没有去尝试。导致s2训练出来的日期数一直在5-15号之间徘徊。
还有人说用obj能不能做，这个还真没考虑过，nn的方法应该也可以，但是由于时间等因素没有去尝试。

