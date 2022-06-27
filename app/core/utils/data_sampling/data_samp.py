import pandas as pd
from imblearn.over_sampling import (
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    ADASYN,
    RandomOverSampler,
    SMOTEN
)


def data_sampling(df, sampling_method, SEED):  # 数据采样
    X = df[df.columns[1:]]
    y = df[df.columns[0]]

    data_sampling_dict = {
        'SMO': SMOTE,
        'SSMO': SVMSMOTE,
        'BSMO': BorderlineSMOTE,
        'ADA': ADASYN,
        'ROS': RandomOverSampler,
        'SMN': SMOTEN
    }

    # 判断标签是否平衡, 不平衡才采样
    from collections import Counter
    counts = Counter(y).most_common()
    if counts[0][1] / len(y) >= 0.8 or counts[-1][1] / len(y) <= 0.2:
        return df

    try:
        X_oversample, y_oversample = data_sampling_dict[sampling_method](
            random_state=SEED
        ).fit_resample(X.values, y.values)
        df1 = pd.DataFrame(X_oversample, columns=df.columns[1:])
        df2 = pd.Series(y_oversample)
        df2.name = df.columns[0]
        df_sampling = pd.concat([df2, df1], axis=1)
        return df_sampling

    except: return df
