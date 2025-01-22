# logit_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
   path('', views.index, name='index'), # ルート URL に views.index を対応させる
]
