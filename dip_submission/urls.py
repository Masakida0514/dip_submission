# path()関数、include()関数のインポート
from django.urls import include, path
# 管理サイトの機能をインポート
from django.contrib import admin

urlpatterns = [
    # iekari アプリケーションの URL 設定を追加
    path('submit/', include('submit_app.urls')),
    # 管理サイト
    path('admin/', admin.site.urls),
]