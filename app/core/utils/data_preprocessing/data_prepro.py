import pandas as pd
import numpy as np

def del_label_line(df):  # 删除标签有缺失的行
    df.iloc[:, 0] = df.iloc[:, 0].fillna('999')
    nan_y = df[(df.iloc[:, 0] == '999')].index.tolist()
    df_del_label = df.drop(nan_y, axis=0)
    print('因标签缺失删除了第{}行的{}个样本'.format(nan_y, len(nan_y)))
    return df_del_label


def del_miss_columns(df):  # 删除缺失值大于80%的列
    df_miss = del_label_line(df)
    nan_ = []
    nan_ra = 0.8
    for i in df_miss.iloc[:, 1:]:
        nan_count = df_miss[i].isnull().sum()
        nan_rate = nan_count/len(df_miss[i])
        nan_rate = (i, nan_rate)
        nan_.append(nan_rate)
    nan_rate = dict(nan_)
    nan_df = pd.Series(nan_rate)
    nan_index = nan_df[nan_df.values > nan_ra].index.tolist()  # 缺失值大于0.8的删除
    df_miss.drop(nan_index, axis=1, inplace=True)
    print('因为缺失值大于{}而删除了{}，共{}个变量'.format(nan_ra, nan_index, len(nan_index)))
    return df_miss


def del_category_colunms(df):  # 删除单个类别大于80%的列
    del_category = del_miss_columns(df)
    x_ = []
    cte_ra = 0.8
    for j in del_category.columns[1:]:
        seis = del_category[j].value_counts()  # 每个变量的个数
        for index_ in seis.index:
            # .isnull()判断是否有缺失值
            cate_rate = seis[index_] / \
                sum(del_category.iloc[:, 1].isnull() == False)

            if cate_rate > cte_ra:  # 删除单个类别比例大于0.8的变量
                del_category.drop(j, axis=1, inplace=True)
                x_.append(j)
    print('因为单个类大于{}而删除了{}，共{}个变量'.format(cte_ra, x_, len(x_)))
    return del_category


def cv_delete(df):  # 删除变异系数小于0.1的列
    df_cv = del_category_colunms(df)
    y_ = []
    cv_ = 0.1
    for k in df_cv.columns[1:]:
        if df_cv[k].mean() != 0:
            cv = df_cv[k].std()/df_cv[k].mean()
        else:
            cv == 10000000

        if cv < cv_:
            df_cv.drop(k, axis=1)
            y_.append(k)
    print('因为变异系数小于{}而删除了{}，共{}个变量'.format(cv_, y_, len(y_)))
    return df_cv


def get_dummis(df, var_type_dict, RESULT_PATH):  # 将多值无序变量设置为哑变量
    mult_disorder = var_type_dict['mult_disorder']
    # df_dumm = cv_delete(df)
    df_dumm = df
    if mult_disorder != ['']:
        for i in mult_disorder:
            if i in df_dumm.columns[0:]:
                df_dummies = pd.get_dummies(
                    df_dumm[i], prefix=i, prefix_sep='_')  # 设置哑变量
                df_dumm.drop(i, inplace=True, axis=1)
                df_dumm = pd.concat([df_dumm, df_dummies], axis=1)

            elif i not in df_dumm.columns[0:]:
                print('多值无序变量{}已被删除'.format(i))
        df_dumm.to_excel(RESULT_PATH + 'tmp/data_pro.xlsx')
        return df_dumm
