import csv
import numpy as np
import pandas as pd
import MeCab
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing, metrics
from django.http import HttpResponse

def process_files(train_x):
    train_x["掲載期間　開始日"] = pd.to_datetime(train_x["掲載期間　開始日"])
    train_x["掲載期間　終了日"] = pd.to_datetime(train_x["掲載期間　終了日"])
    train_x["期間・時間　勤務開始日"] = pd.to_datetime(train_x["期間・時間　勤務開始日"])

    nlp_list = ["休日休暇　備考", "（紹介予定）入社時期", "お仕事名", "（派遣先）配属先部署", "仕事内容", "（派遣先）概要　事業内容",
                "（紹介予定）年収・給与例", "応募資格", "（紹介予定）休日休暇", "派遣会社のうれしい特典", "お仕事のポイント（仕事PR）", "（派遣先）職場の雰囲気", "（紹介予定）待遇・福利厚生",
                "給与/交通費　備考", "期間･時間　備考"]

    def split_text_only_noun(text):
        tagger = MeCab.Tagger()

        words = []
        for c in tagger.parse(text).splitlines()[:-1]:
            surface, feature = c.split('\t')
            pos = feature.split(',')[0]
            if pos == '名詞':
                words.append(surface)
        return ' '.join(words)

    train_x["mecab"] = train_x[nlp_list[0]].fillna('')
    nlp_list.remove("休日休暇　備考")
    for nlp in nlp_list:
        train_x["mecab"] = train_x["mecab"] + ' ' + train_x[nlp].fillna('')
        train_x.drop(nlp, axis=1, inplace=True)

    train_x.drop("休日休暇　備考", axis=1, inplace=True)
    train_x["mecab"] = (train_x["mecab"].apply(split_text_only_noun))

    def count_vect(df):
        df = np.array(df)
        count = CountVectorizer()
        count.fit(df)
        df = count.transform(df)
        tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
        tfidf.fit(df)
        df = tfidf.transform(df)
        df = pd.DataFrame(df.toarray(), columns=count.get_feature_names())
        return df

    bags = count_vect(train_x["mecab"])
    bags = pd.DataFrame(bags)

    train_x.drop("mecab", axis=1, inplace=True)
    train_x = pd.concat([train_x, bags], axis=1)

    del bags
    del words_list

    work_time = train_x["期間・時間　勤務時間"].str.split('<BR>')
    work_time = work_time.apply(pd.Series)
    work_time.columns = ['時間', '残業', '休憩', 'nazo']
    work_time = work_time['時間'].str.replace('\u3000', '')
    work_time = work_time.str.split('〜')
    work_time = work_time.apply(pd.Series)
    work_time.columns = ['始業', '終業']

    def pandas_datetime(df):
        df = df.split(':')
        df = datetime.time(int(df[0]), int(df[1]))
        return df

    work_time = work_time.applymap(pandas_datetime)

    def make_datetime(df):
        birth_day = datetime.datetime(2000, 5, 14)
        birth_day = datetime.datetime.combine(birth_day, df)
        return birth_day

    work_time['tmp1'] = work_time['始業'].apply(make_datetime)
    work_time['tmp2'] = work_time['終業'].apply(make_datetime)
    work_time["拘束時間"] = work_time['tmp2'] - work_time['tmp1']
    work_time.drop(["tmp1", "tmp2"], axis=1, inplace=True)

    train_x.drop("期間・時間　勤務時間", axis=1, inplace=True)

    train_x = pd.concat([train_x, work_time], axis=1)

    train_x["（紹介予定）雇用形態備考"].fillna('正社員', inplace=True)
    tmp = train_x["（派遣先）勤務先写真ファイル名"]
    tmp[tmp.notnull()] = 1
    tmp.fillna(0, inplace=True)
    train_x["（派遣先）勤務先写真ファイル名"] = tmp

    def transform(data):
        cat = ["勤務地　最寄駅2（駅名）",
               "（紹介予定）雇用形態備考",
               "勤務地　最寄駅2（沿線名）",
               "（派遣先）概要　勤務先名（漢字）",
               "勤務地　備考",
               "拠点番号",
               "勤務地　最寄駅1（沿線名）",
               "勤務地　最寄駅1（駅名）"]

        for feature in cat:
            data[feature].fillna('unknown', inplace=True)
            encoder = preprocessing.LabelEncoder()
            data[feature] = encoder.fit_transform(data[feature])
        return data

    label_encoder = train_x.select_dtypes(include='object')
    label_encoder = transform(label_encoder)

    categorical_variable = ["勤務地　最寄駅2（駅名）",
                            "（紹介予定）雇用形態備考",
                            "勤務地　最寄駅2（沿線名）",
                            "（派遣先）概要　勤務先名（漢字）",
                            "勤務地　備考",
                            "拠点番号",
                            "勤務地　最寄駅1（沿線名）",
                            "勤務地　最寄駅1（駅名）"]
    for attr in categorical_variable:
        train_x["{}".format(attr)] = label_encoder["{}".format(attr)]

    sigyou = np.where(train_x.columns.get_loc('始業'))
    sigyou = sigyou.tolist()
    syuugyou = np.where(train_x.columns.get_loc('終業'))
    syuugyou = syuugyou.tolist()

    if len(sigyou) != 1 and len(syuugyou) != 1:
        try:
            tmp_1 = train_x.iloc[:, [sigyou[0], syuugyou[0]]].rename(columns={'始業': '始業_Mecab', '終業': '終業_Mecab'})
            tmp_2 = train_x.iloc[:, [sigyou[1], syuugyou[1]]]
            train_x.drop(['始業', '終業'], inplace=True, axis=1)
            train_x = pd.concat([train_x, tmp_1, tmp_2], axis=1)
        except:
            pass

    tmp_1 = ['掲載期間　開始日', '期間・時間　勤務開始日', '掲載期間　終了日']
    tmp_2 = ['始業', '終業']

    def make_num(col):
        train_x[col + "year"] = train_x[col].map(lambda x: x.year)
        train_x[col + "month"] = train_x[col].map(lambda x: x.month)
        train_x[col + "day"] = train_x[col].map(lambda x: x.day)

    for i in tmp_1:
        make_num(i)

    def make_num_2(col):
        train_x[col + "hour"] = train_x[col].apply(lambda x: x.hour)
        train_x[col + "minute"] = train_x[col].map(lambda x: x.minute)

    for i in tmp_2:
        make_num_2(i)

    def make_num_3(col):
        train_x[col + "total_seconds"] = train_x[col].map(lambda x: x.total_seconds())

    make_num_3('拘束時間')

    columns = tmp_1 + tmp_2
    columns.append('拘束時間')
    data = train_x.drop(columns, axis=1)

    return data

def model_loader(path):
    filename = '505_sklearn_randomforest.sav'
    filename = path + filename
    random_forest = pickle.load(open(filename, 'rb'))

    return random_forest

def model_fitter(random_forest, data):
    y_pred = random_forest.predict(data)
    submission = pd.DataFrame(y_pred)
    submission = pd.concat([submission, data['お仕事No.']], axis=1)
    submission.rename(columns={0: '応募数 合計'}, inplace=True)
    submission.set_index('お仕事No.', inplace=True)

    return  submission

def write_into_csv(csv_data):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="download.csv"'
    csv_data.to_csv(path_or_buf=response)
    #, sep=';', float_format='%.2f', index=False,decimal=",")


    return response

if __name__ == '__main__':
    path = "static/mode/"

    recruitment_info = pd.read_csv("static/test/test_x.csv")
    recruitment_info = process_files(recruitment_info)
    random_forest = model_loader(path)
    recruitment_pred = model_fitter(random_forest, recruitment_info)
    # response = write_into_csv(recruitment_pred)

    print(recruitment_pred)