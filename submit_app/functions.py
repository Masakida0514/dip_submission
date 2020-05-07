import pandas as pd
from django.http import HttpResponse

from submit_app.utils import (
    reduce_mem_usage,
    split_text_only_noun,
    count_vect, pandas_datetime,
    make_datetime,
    transform,
    transform_columns
)


def process_files(train_x):
    # 前処理用に入力ファイルのカラムを整理
    train_x = reduce_mem_usage(train_x)
    train_x_columns = pd.read_pickle('submit_app/static/columns/train_x_columns.pickle')
    train_x = train_x[train_x_columns.columns]
    del train_x_columns

    train_x["掲載期間　開始日"] = pd.to_datetime(train_x["掲載期間　開始日"])
    train_x["掲載期間　終了日"] = pd.to_datetime(train_x["掲載期間　終了日"])
    train_x["期間・時間　勤務開始日"] = pd.to_datetime(train_x["期間・時間　勤務開始日"])

    # 自然言語処理が必要なリスト
    nlp_list = ["休日休暇　備考", "（紹介予定）入社時期", "お仕事名", "（派遣先）配属先部署", "仕事内容", "（派遣先）概要　事業内容",
                "（紹介予定）年収・給与例", "応募資格", "（紹介予定）休日休暇", "派遣会社のうれしい特典", "お仕事のポイント（仕事PR）", "（派遣先）職場の雰囲気", "（紹介予定）待遇・福利厚生",
                "給与/交通費　備考", "期間･時間　備考"]

    # 'mecab'カラムにnlp_list内の言葉を全て入れる
    train_x["mecab"] = train_x[nlp_list[0]].fillna('')
    nlp_list.remove("休日休暇　備考")
    for nlp in nlp_list:
        train_x["mecab"] = train_x["mecab"] + ' ' + train_x[nlp].fillna('')
        train_x.drop(nlp, axis=1, inplace=True)
    train_x.drop("休日休暇　備考", axis=1, inplace=True)

    # 'mecab'カラムを名詞だけを入れたカラムに更新
    train_x["mecab"] = (train_x["mecab"].apply(split_text_only_noun))

    # bagsに名詞の登場回数を格納
    bags = count_vect(train_x["mecab"])
    bags = pd.DataFrame(bags)

    # 'mecab'カラムの削除
    train_x.drop("mecab", axis=1, inplace=True)

    # 時系列処理
    work_time = train_x["期間・時間　勤務時間"].str.split('<BR>')
    work_time = work_time.apply(pd.Series)
    work_time.columns = ['時間', '残業', '休憩', 'nazo']
    work_time = work_time['時間'].str.replace('\u3000', '')
    work_time = work_time.str.split('〜')
    work_time = work_time.apply(pd.Series)
    work_time.columns = ['始業', '終業']

    # datetime型に変換
    work_time = work_time.applymap(pandas_datetime)

    # timedeltaを計算するために、年月日情報を追加
    work_time['tmp1'] = work_time['始業'].apply(make_datetime)
    work_time['tmp2'] = work_time['終業'].apply(make_datetime)
    work_time["拘束時間"] = work_time['tmp2'] - work_time['tmp1']
    work_time.drop(["tmp1", "tmp2"], axis=1, inplace=True)
    train_x.drop("期間・時間　勤務時間", axis=1, inplace=True)

    train_x = pd.concat([train_x, work_time], axis=1)

    # カテゴリ変数化処理
    # 欠損値補完
    train_x["（紹介予定）雇用形態備考"].fillna('正社員', inplace=True)

    # 写真があるかないかのダミー変数
    tmp = train_x["（派遣先）勤務先写真ファイル名"]
    tmp[tmp.notnull()] = 1
    tmp.fillna(0, inplace=True)
    train_x["（派遣先）勤務先写真ファイル名"] = tmp

    # カテゴリ変数化処理したtrain_xとダミー変数化したdummiesを得る
    train_x, dummies = transform(train_x)

    # 時系列処理
    tmp_1 = ['掲載期間　開始日', '期間・時間　勤務開始日', '掲載期間　終了日']
    tmp_2 = ['始業', '終業']

    def make_num(col):
        train_x[col + "year"] = train_x[col].map(lambda x: x.year)
        train_x[col + "month"] = train_x[col].map(lambda x: x.month)
        train_x[col + "day"] = train_x[col].map(lambda x: x.day)

    def make_num_2(col):
        train_x[col + "hour"] = train_x[col].apply(lambda x: x.hour)
        train_x[col + "minute"] = train_x[col].map(lambda x: x.minute)

    def make_num_3(col):
        train_x[col + "total_seconds"] = train_x[col].map(lambda x: x.total_seconds())

    for i in tmp_1:
        make_num(i)

    for i in tmp_2:
        make_num_2(i)

    make_num_3('拘束時間')

    columns = tmp_1 + tmp_2
    columns.append('拘束時間')
    data = train_x.drop(columns, axis=1)

    # dataとbagsとdummiesのカラムを解析用に処理+連結
    data = transform_columns(data, bags, dummies)

    return data


def model_loader():
    path = 'submit_app/static/model/'
    filename = '506_sklearn_randomforest.sav'
    filename = path + filename
    random_forest = pickle.load(open(filename, 'rb'))

    return random_forest


def model_fitter(random_forest, data):
    y_pred = random_forest.predict(data)
    submission = pd.DataFrame(y_pred)
    submission = pd.concat([submission, data['お仕事No.']], axis=1)
    submission.rename(columns={0: '応募数 合計'}, inplace=True)
    submission.set_index('お仕事No.', inplace=True)

    return submission


def write_into_csv(csv_data):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="predict_y.csv"'
    csv_data.to_csv(path_or_buf=response)
    # , sep=';', float_format='%.2f', index=False,decimal=",")

    return response


# if __name__ == '__main__':
#     path = "submit_app/static/"
#
#     recruitment_info = pd.read_csv("test/test_x.csv")
#     recruitment_info = process_files(recruitment_info)
#     random_forest = model_loader(path)
#     recruitment_pred = model_fitter(random_forest, recruitment_info)
#     # response = write_into_csv(recruitment_pred)
#
#     print(recruitment_pred)

