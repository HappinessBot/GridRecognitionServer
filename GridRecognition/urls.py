from django.conf.urls import url

from GridRecognition import views

urlpatterns = [
    url(r'^matrix/', views.get_matrix),
]
