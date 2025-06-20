from django.urls import path
from .views import hello_world
from .views import latest_tickets

urlpatterns = [
    path('', hello_world, name='hello_world'),
    path('latest-tickets/', latest_tickets, name='latest_tickets'),
]