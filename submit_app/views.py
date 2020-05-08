from django.shortcuts import render
import pandas as pd
from submit_app.forms import RecruitmentForm
from submit_app.functions import process_files, model_loader, model_fitter, write_into_csv

import logging


def index(request):
    # djangoにおいてファイルのPOSTは、request.FILESに格納される。
    # このためHTMLのformのタグには、enctype="multipart/form-data"を記述すること
    # request.FILESは、dictオブジェクト。
    # input type='files' name='name'　のname部がdictのキーになる

    if request.method == 'POST':
        recruitment = RecruitmentForm(request.POST, request.FILES)
        if recruitment.is_valid():
            # ログレベルを DEBUG に変更
            logging.basicConfig(level=logging.DEBUG)

            recruitment_info = pd.read_csv(request.FILES['recruitment_info'])
            logging.info('{}'.format('read files'))
            recruitment_info, work_num = process_files(recruitment_info)
            logging.info('{}'.format('processed files'))
            random_forest = model_loader()
            logging.info('{}'.format('load models'))
            recruitment_pred = model_fitter(random_forest, recruitment_info, work_num)
            logging.info('{}'.format('fit models'))
            response = write_into_csv(recruitment_pred)
            logging.info('{}'.format('write_csv'))

            return response

    else:
        recruitment = RecruitmentForm()
        return render(request, 'index.html', {'form': recruitment})
