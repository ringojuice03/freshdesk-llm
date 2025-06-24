from django.urls import path
from .views import hello_world
from .views import latest_tickets, embed_tickets, get_payload_to_json

urlpatterns = [
    path('', hello_world, name='hello_world'),
    path('latest-tickets/', latest_tickets, name='latest_tickets'),
    path('embed-tickets/', embed_tickets, name='embed_tickets'),
    path('json/', get_payload_to_json, name='to_json'),
]