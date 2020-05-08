import numpy as np
import pandas as pd
import datetime
import logging

import MeCab
from sklearn.feature_extraction.text import CountVectorizer


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def split_text_only_noun(text):
    tagger = MeCab.Tagger()

    words = []
    for c in tagger.parse(text).splitlines()[:-1]:
        surface, feature = c.split('\t')
        pos = feature.split(',')[0]
        if pos == '名詞':
            words.append(surface)
    return ' '.join(words)


def count_vect(df):
    df = np.array(df)
    count = CountVectorizer()
    count.fit(df)
    df = count.transform(df)
    df = pd.DataFrame(df.toarray(), columns=count.get_feature_names())
    return df


def pandas_datetime(df):
    df = df.split(':')
    df = datetime.time(int(df[0]), int(df[1]))
    return df


def make_datetime(df):
    birth_day = datetime.datetime(2000, 5, 14)
    birth_day = datetime.datetime.combine(birth_day, df)
    return birth_day


'''
transform関数
カテゴリ変数化　→　ランダムフォレストではlabelencoding可能だが、
入力データの前処理の際にずれる可能性があるため、one-hot-encodingにする。
'''


def transform(data):
    cat = ["勤務地　最寄駅2（駅名）", "（紹介予定）雇用形態備考", "勤務地　最寄駅2（沿線名）", "（派遣先）概要　勤務先名（漢字）",
           "勤務地　備考", "拠点番号", "勤務地　最寄駅1（沿線名）", "勤務地　最寄駅1（駅名）"]
    # 欠損値の補完
    data[cat].fillna('unknown', inplace=True)
    # ダミー変数取得
    data = pd.get_dummies(data, columns=cat)

    return data


def transform_columns(train_x, bags):
    def common_searcher(df, path):
        columns = pd.read_pickle(path)
        columns_and = set(list(df.columns)) & set(list(columns.columns))
        columns_rest = set(list(columns_and)) ^ set(list(columns.columns))
        df = df[columns_and]
        for column in list(columns_rest):
            df[column] = 0
        return df

    path = 'submit_app/static/columns/508_train_x_processed.pickle'
    train_x = common_searcher(train_x, path)
    path = 'submit_app/static/columns/508_bags.pickle'
    bags = common_searcher(bags, path)

    train_x = pd.concat([train_x, bags], axis=1)
    train_x.fillna(0, inplace=True)
    return train_x
