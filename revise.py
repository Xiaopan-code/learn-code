import pickle
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
# 数据读取
dfoff = pd.read_csv('ccf_offline_stage1_train.csv')
dftest = pd.read_csv('ccf_offline_stage1_test_revised.csv')
dfon = pd.read_csv('ccf_online_stage1_train.csv')

# 数据预处理函数
def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0


def convertRate(row):
    """Convert discount to rate"""
    if pd.isnull(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)


def getDiscountMan(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


def getDiscountJian(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


def processData(df):
    # 转换折扣率相关特征
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    # 转换距离特征
    df['distance'] = df['Distance'].fillna(-1).astype(int)
    return df


# 处理折扣和距离特征
dfoff = processData(dfoff)
dftest = processData(dftest)

# 提取日期特征
date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[pd.notnull(date_received)])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[pd.notnull(date_buy)])
date_buy = sorted(dfoff[dfoff['Date'].notnull()]['Date'])

# 每天领取优惠券数量（修正agg参数格式）
couponbydate = dfoff[dfoff['Date_received'].notnull()][['Date_received', 'Date']].groupby(
    ['Date_received'], as_index=False)['Date'].agg(count=np.size)
couponbydate.columns = ['Date_received', 'count']

# 每天消耗优惠券数量
buybydate = dfoff[(dfoff['Date'].notnull()) & (dfoff['Date_received'].notnull())][
    ['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received', 'count']

# 用户优惠券消费次数特征（修正fillna方式）
temp_user_coupon = dfoff[(dfoff['Date'].notnull()) & (dfoff['Date_received'].notnull())]
user_coupon = temp_user_coupon.groupby(['User_id']).size().reset_index(name='user_coupon')

dfoff = pd.merge(dfoff, user_coupon, how='left', on='User_id')
dfoff['user_coupon'] = dfoff['user_coupon'].fillna(0)  # 替代inplace=True
dftest = pd.merge(dftest, user_coupon, how='left', on='User_id')
dftest['user_coupon'] = dftest['user_coupon'].fillna(0)  # 替代inplace=True

# 优惠券消费最大间隔（修正agg参数和函数引用）
date_coupon = temp_user_coupon.groupby('User_id', as_index=False).Date.agg(cmax='max', cmin='min')  # 使用字符串函数名
date_coupon[['cmin', 'cmax']] = date_coupon[['cmin', 'cmax']].astype('int').astype('str')
date_coupon['cmax'] = date_coupon['cmax'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
date_coupon['cmin'] = date_coupon['cmin'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
date_coupon['cdate_interval'] = (date_coupon['cmax'] - date_coupon['cmin']).dt.days

dfoff = pd.merge(dfoff, date_coupon, how='left', on='User_id')
dftest = pd.merge(dftest, date_coupon, how='left', on='User_id')
dfoff['cdate_interval'] = dfoff['cdate_interval'].fillna(-1)
dftest['cdate_interval'] = dftest['cdate_interval'].fillna(-1)

# 普通消费次数特征（修正fillna方式）
temp_user_nocoupon = dfoff[(dfoff['Date'].notnull()) & (dfoff['Date_received'].isnull())]
user_nocoupon = temp_user_nocoupon.groupby(['User_id']).size().reset_index(name='user_nocoupon')

dfoff = pd.merge(dfoff, user_nocoupon, how='left', on='User_id')
dfoff['user_nocoupon'] = dfoff['user_nocoupon'].fillna(0)  # 替代inplace=True
dftest = pd.merge(dftest, user_nocoupon, how='left', on='User_id')
dftest['user_nocoupon'] = dftest['user_nocoupon'].fillna(0)  # 替代inplace=True

# 普通消费最大间隔（修正agg参数和函数引用）
date1 = temp_user_nocoupon.groupby('User_id', as_index=False).Date.agg(max='max', min='min')  # 使用字符串函数名
date1[['min', 'max']] = date1[['min', 'max']].astype('int').astype('str')
date1['max'] = date1['max'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
date1['min'] = date1['min'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
date1['date_interval'] = (date1['max'] - date1['min']).dt.days

dfoff = pd.merge(dfoff, date1, how='left', on='User_id')
dftest = pd.merge(dftest, date1, how='left', on='User_id')
dfoff['date_interval'] = dfoff['date_interval'].fillna(-1)
dftest['date_interval'] = dftest['date_interval'].fillna(-1)

# 星期特征处理
def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

dfoff['weekday'] = dfoff['Date_received'].astype('str').apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype('str').apply(getWeekday)

# 周末标识特征
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)

# 星期特征one-hot编码
weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

# 标签定义（是否在15天内使用优惠券）
def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0

dfoff['label'] = dfoff.apply(label, axis=1)

# 数据拆分
df = dfoff[dfoff['label'] != -1].copy()
df.to_csv('df.csv', index=False)

train = df[(df['Date_received'] < 20160516)].copy()
valid = df[(df['Date_received'] >= 20160516) & (df['Date_received'] <= 20160615)].copy()

# 特征定义
df = pd.read_csv('df.csv')
original_feature = ['discount_rate', 'discount_man',
                    'discount_jian', 'discount_type', 'distance', 'user_coupon', 'weekday',
                    'weekday_type', 'user_nocoupon', 'date_interval'] + weekdaycols

# 随机森林模型训练与调优
param_test = {
    'max_depth': range(3, 5, 1),
    'n_estimators': range(150, 200, 30)
}
gsearch = GridSearchCV(
    estimator=RandomForestClassifier(criterion='gini'),
    param_grid=param_test,
    cv=KFold(2),
    scoring='roc_auc'
)
gsearch.fit(train[original_feature], train['label'])

rf_model = gsearch.best_estimator_

# 模型评估
rf_pre_train = rf_model.predict_proba(train[original_feature])
rf_auc_train = roc_auc_score(train['label'], rf_pre_train[:, 1])
rf_pre_test = rf_model.predict_proba(valid[original_feature])
rf_auc_valid = roc_auc_score(valid['label'], rf_pre_test[:, 1])
print('rf_auc_train: ', rf_auc_train)
print('rf_auc_valid: ', rf_auc_valid)

# 保存模型
print("---save model---")
with open('RF.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# 测试集预测与提交文件生成
y_test_pred = rf_model.predict_proba(dftest[original_feature])
dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
dftest1['label'] = y_test_pred[:, 1]
dftest1.to_csv('RFsubmit.csv', index=False, header=False)
print(dftest1.head())