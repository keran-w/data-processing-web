import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, SMOTEN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor


def feature_selection(df, sele_method, SEED):  # 特征筛选
    X = df[df.columns[1:]].values
    y = df[df.columns[0]].values
    imp_dict = {}
    standardizer = StandardScaler()
    X_standardized = standardizer.fit_transform(X)

    if sele_method == 'Las':
        clf = LassoCV(random_state=SEED)
        clf.fit(X_standardized, y)
        feature_importance = clf.coef_[0]
        feature_importance = np.array(feature_importance)
        feature_importance = (feature_importance ** 2) ** 0.5
        feature_importance = feature_importance / np.sum(feature_importance)

        imp_dict[sele_method] = feature_importance
        fea_importance_pd = pd.DataFrame(imp_dict, index=[0])
        fea_series = pd.Series(df.columns[1:])
        fea_importance_df = pd.concat([fea_series, fea_importance_pd], axis=1)

        df_sort = fea_importance_df.sort_values(
            by=sele_method, ascending=False)
        variable_selected = list(df_sort.iloc[:10, 0])

    elif sele_method == 'RCV':
        clf = RidgeCV()
        clf.fit(X_standardized, y)
        feature_importance = clf.coef_
        feature_importance = np.array(feature_importance)
        feature_importance = (feature_importance ** 2) ** 0.5
        feature_importance = feature_importance / np.sum(feature_importance)

        imp_dict[sele_method] = feature_importance
        fea_importance_pd = pd.DataFrame(imp_dict)
        fea_series = pd.Series(df.columns[1:])
        fea_importance_df = pd.concat([fea_series, fea_importance_pd], axis=1)

        df_sort = fea_importance_df.sort_values(
            by=sele_method, ascending=False)
        variable_selected = list(df_sort.iloc[:10, 0])

    elif sele_method == 'ENC':
        clf = ElasticNetCV(random_state=SEED)
        clf.fit(X_standardized, y)
        feature_importance = clf.coef_
        feature_importance = np.array(feature_importance)
        feature_importance = (feature_importance ** 2) ** 0.5
        feature_importance = feature_importance / np.sum(feature_importance)

        imp_dict[sele_method] = feature_importance
        fea_importance_pd = pd.DataFrame(imp_dict)
        fea_series = pd.Series(df.columns[1:])
        fea_importance_df = pd.concat([fea_series, fea_importance_pd], axis=1)

        df_sort = fea_importance_df.sort_values(
            by=sele_method, ascending=False)
        variable_selected = list(df_sort.iloc[:10, 0])

    elif sele_method == 'Cat':
        clf = CatBoostClassifier(random_state=SEED, logging_level='Silent')
        clf.fit(X_standardized, y)
        feature_importance = clf.feature_importances_
        feature_importance = feature_importance / np.sum(feature_importance)

        imp_dict[sele_method] = feature_importance
        fea_importance_pd = pd.DataFrame(imp_dict)
        fea_series = pd.Series(df.columns[1:])
        fea_importance_df = pd.concat([fea_series, fea_importance_pd], axis=1)

        df_sort = fea_importance_df.sort_values(
            by=sele_method, ascending=False)
        variable_selected = list(df_sort.iloc[:10, 0])

    elif sele_method == 'SVC':
        clf = SVC(kernel='linear', random_state=SEED)
        clf.fit(X_standardized, y)
        feature_importance = clf.coef_[0]
        feature_importance = np.array(feature_importance)
        feature_importance = (feature_importance ** 2) ** 0.5  # 变成正数
        feature_importance = feature_importance / np.sum(feature_importance)

        imp_dict[sele_method] = feature_importance
        fea_importance_pd = pd.DataFrame(imp_dict)
        fea_series = pd.Series(df.columns[1:])
        fea_importance_df = pd.concat([fea_series, fea_importance_pd], axis=1)

        df_sort = fea_importance_df.sort_values(
            by=sele_method, ascending=False)
        variable_selected = list(df_sort.iloc[:10, 0])

    elif sele_method == 'RF':
        ranking_ = {}
        estimator = RandomForestClassifier(random_state=SEED)
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
        rfecv.fit(X_standardized, y.astype('int'))
        ranking_[sele_method] = rfecv.ranking_
        ranking_df = pd.DataFrame.from_dict(ranking_)  # 将Dict转换为DataFrame对象
        ranking_df['Variable'] = df.columns[1:]
        ranking_df.set_index(['Variable'], inplace=True)

        variable_selected = list(
            ranking_df[ranking_df[sele_method] == 1].index)

    elif sele_method == 'Ada':
        ranking_ = {}
        estimator = AdaBoostClassifier(random_state=SEED)
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
        rfecv.fit(X_standardized, y.astype('int'))
        ranking_[sele_method] = rfecv.ranking_
        ranking_df = pd.DataFrame.from_dict(ranking_)  # 将Dict转换为DataFrame对象
        ranking_df['Variable'] = df.columns[1:]
        ranking_df.set_index(['Variable'], inplace=True)

        variable_selected = list(
            ranking_df[ranking_df[sele_method] == 1].index)

    elif sele_method == 'GBC':
        ranking_ = {}
        estimator = GradientBoostingClassifier(random_state=SEED)
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
        rfecv.fit(X_standardized, y.astype('int'))
        ranking_[sele_method] = rfecv.ranking_
        ranking_df = pd.DataFrame.from_dict(ranking_)  # 将Dict转换为DataFrame对象
        ranking_df['Variable'] = df.columns[1:]
        ranking_df.set_index(['Variable'], inplace=True)

        variable_selected = list(
            ranking_df[ranking_df[sele_method] == 1].index)

    elif sele_method == 'ExT':
        ranking_ = {}
        estimator = ExtraTreesClassifier(random_state=SEED)
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
        rfecv.fit(X_standardized, y.astype('int'))
        ranking_[sele_method] = rfecv.ranking_
        ranking_df = pd.DataFrame.from_dict(ranking_)  # 将Dict转换为DataFrame对象
        ranking_df['Variable'] = df.columns[1:]
        ranking_df.set_index(['Variable'], inplace=True)

        variable_selected = list(
            ranking_df[ranking_df[sele_method] == 1].index)

    elif sele_method == 'BNB':
        ranking_ = {}
        estimator = BernoulliNB()
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
        rfecv.fit(X_standardized, y.astype('int'))
        ranking_[sele_method] = rfecv.ranking_
        ranking_df = pd.DataFrame.from_dict(ranking_)  # 将Dict转换为DataFrame对象
        ranking_df['Variable'] = df.columns[1:]
        ranking_df.set_index(['Variable'], inplace=True)

        variable_selected = list(
            ranking_df[ranking_df[sele_method] == 1].index)

    elif sele_method == 'XGB':
        ranking_ = {}
        estimator = XGBClassifier()
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
        rfecv.fit(X_standardized, y.astype('int'))
        ranking_[sele_method] = rfecv.ranking_
        ranking_df = pd.DataFrame.from_dict(ranking_)  # 将Dict转换为DataFrame对象
        ranking_df['Variable'] = df.columns[1:]
        ranking_df.set_index(['Variable'], inplace=True)

        variable_selected = list(
            ranking_df[ranking_df[sele_method] == 1].index)

    elif sele_method == 'LGBM':
        ranking_ = {}
        estimator = LGBMClassifier(random_state=SEED)
        rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='roc_auc')
        rfecv.fit(X_standardized, y.astype('int'))
        ranking_[sele_method] = rfecv.ranking_
        ranking_df = pd.DataFrame.from_dict(ranking_)  # 将Dict转换为DataFrame对象
        ranking_df['Variable'] = df.columns[1:]
        ranking_df.set_index(['Variable'], inplace=True)

        variable_selected = list(
            ranking_df[ranking_df[sele_method] == 1].index)

    return variable_selected
