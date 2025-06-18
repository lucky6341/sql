# urls.py - Complete URL configuration with all API endpoints
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import (
    DatasetViewSet, QuestionViewSet, UserAttemptViewSet,
    AchievementViewSet, LearningPathViewSet
)

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'questions', QuestionViewSet, basename='question')
router.register(r'attempts', UserAttemptViewSet, basename='attempt')
router.register(r'achievements', AchievementViewSet, basename='achievement')
router.register(r'learning-paths', LearningPathViewSet, basename='learningpath')

urlpatterns = [
    # API Endpoints
    path('api/', include(router.urls)),
    
    # Dataset Management
    path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    path('datasets/upload/', views.DatasetUploadView.as_view(), name='dataset_upload'),
    path('datasets/<uuid:pk>/', views.DatasetDetailView.as_view(), name='dataset_detail'),
    path('datasets/<uuid:dataset_id>/rate/', views.dataset_rate, name='dataset_rate'),
    path('datasets/<uuid:dataset_id>/questions/', views.dataset_questions, name='dataset_questions'),
    
    # Batch Processing
    path('api/datasets/batch-process/', views.process_csv_folder, name='batch_process'),
    
    # Question Execution
    path('questions/<uuid:pk>/', views.QuestionDetailView.as_view(), name='question_detail'),
    path('questions/<uuid:question_id>/execute/', views.execute_query, name='execute_query'),
    path('questions/<uuid:question_id>/hint/', views.get_hint, name='get_hint'),
    path('questions/<uuid:question_id>/solution/', views.view_solution, name='view_solution'),
    
    # AI Features
    path('api/ai/generate-questions/', views.generate_practice_questions, name='generate_questions'),
    path('api/ai/explain-query/', views.explain_query, name='explain_query'),
    path('api/ai/feedback/', views.get_ai_feedback, name='ai_feedback'),
    
    # User Progress
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('profile/', views.UserProfileView.as_view(), name='profile'),
    path('progress/', views.UserProgressView.as_view(), name='progress'),
    path('leaderboard/', views.LeaderboardView.as_view(), name='leaderboard'),
    
    # Learning Paths
    path('learning-paths/', views.LearningPathListView.as_view(), name='learningpath_list'),
    path('learning-paths/<uuid:pk>/', views.LearningPathDetailView.as_view(), name='learningpath_detail'),
    path('learning-paths/<uuid:path_id>/start/', views.start_learning_path, name='start_learning_path'),
    
    # Authentication
    path('accounts/', include('django.contrib.auth.urls')),
    
    # Admin
    path('admin/datasets/export/', views.export_datasets, name='export_datasets'),
    
    # Error Handling
    path('400/', views.bad_request, name='bad_request'),
    path('403/', views.permission_denied, name='permission_denied'),
    path('404/', views.page_not_found, name='page_not_found'),
    path('500/', views.server_error, name='server_error'),
]

handler400 = 'techysql.views.bad_request'
handler403 = 'techysql.views.permission_denied'
handler404 = 'techysql.views.page_not_found'
handler500 = 'techysql.views.server_error'