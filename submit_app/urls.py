# path()関数のインボート
from django.urls import path
from . import views


app_name = 'submit_app'
# ルーティングの設定
urlpatterns = [
    # path('regression', regression.form_view, name='submit_form')
    # path('regression/result', regression.result_view, name='regression_result')
    path('', views.index, name='index_test')
]