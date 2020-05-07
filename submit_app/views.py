from django.shortcuts import render
import pandas as pd
from submit_app.forms import RecruitmentForm
from submit_app.functions import process_files, model_loader, model_fitter, write_into_csv


def index(request):
    # djangoにおいてファイルのPOSTは、request.FILESに格納される。
    # このためHTMLのformのタグには、enctype="multipart/form-data"を記述すること
    # request.FILESは、dictオブジェクト。
    # input type='files' name='name'　のname部がdictのキーになる

    if request.method == 'POST':
        recruitment = RecruitmentForm(request.POST, request.FILES)
        if recruitment.is_valid():
            recruitment_info = pd.read_csv(request.FILES['recruitment_info'])
            recruitment_info = process_files(recruitment_info)
            random_forest = model_loader()
            recruitment_pred = model_fitter(random_forest, recruitment_info)
            response = write_into_csv(recruitment_pred)

            return response

    else:
        recruitment = RecruitmentForm()
        return render(request, 'index.html', {'form': recruitment})
