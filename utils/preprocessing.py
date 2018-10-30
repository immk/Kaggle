import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold

def check_for_missing_value(df):
    missing_list = []

    for col in df.columns:
        missing = df[df[col] == -1][col].count()
        if missing > 0:
            missing_list.append(col)
    print("[Missing_Value] {}".format(missing_list))
    return missing_list


def preprocessing_for_missing_value(df):
    missing_list = check_for_missing_value(df)

    mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
    mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

    for col in missing_list:
        if len(df[df[col] == -1]) < 30:
            df[col] = mode_imp.fit_transform(df[[col]]).ravel()
        else:
            df[col] = mean_imp.fit_transform(df[[col]]).ravel()
    print("[Missing_Value Imputation] {}".format(df.shape))
    return df


def preprocessing_for_catefory(df):
    for col in df.columns:
        if 'cat' in col:
            temp_df = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, temp_df], axis=1)
            df.drop(columns=col, inplace=True)

    print("[One-Hot Encoding] {}".format(df.shape))
    return df


def preprocessing_for_drop_feature(df, drop_feature):
    df.drop(columns=drop_feature, axis=1, inplace=True)
    return df


def preprocessing_for_variance(df, th=.01):
    var = VarianceThreshold(threshold=th)
    return var.fit_transform(df)

