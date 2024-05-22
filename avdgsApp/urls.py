# urls.py

from django.urls import path
from avdgsApp import views

urlpatterns = [
    path('', views.home, name='home'),
    path('camera_feed/', views.camera_feed, name='camera_feed'),
    # path('camera_frame_stream/', views.camera_frame_stream, name='camera_frame_stream'),
]
