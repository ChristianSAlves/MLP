from django.urls import path
from .views import MLPTrainAPIView, MLPredictAPIView

urlpatterns = [
    path('train/', MLPTrainAPIView.as_view(), name='mlp-train'),
    path('predict/', MLPredictAPIView.as_view(), name='mlp-predict'),
]
