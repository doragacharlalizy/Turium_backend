from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ContentViewSet, ChunkViewSet, QueryViewSet

# Create router
router = DefaultRouter()

# Register ViewSets
router.register(r'content', ContentViewSet, basename='content')
router.register(r'chunks', ChunkViewSet, basename='chunk')
router.register(r'query', QueryViewSet, basename='query')

# URL patterns
urlpatterns = [
    path('', include(router.urls)),
]
