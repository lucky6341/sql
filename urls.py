# urls.py - Complete URL Configuration for TechySQL Academy
from django.urls import path, include
from django.contrib import admin
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

app_name = 'sql_code_editor'

urlpatterns = [
    # Home and Main Pages
    path('', views.HomeView.as_view(), name='sql_home'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    
    # API Endpoints
    path('api/', include(router.urls)),
    
    # Dataset Management
    path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    path('datasets/upload/', views.DatasetUploadView.as_view(), name='dataset_upload'),
    path('datasets/<uuid:pk>/', views.DatasetDetailView.as_view(), name='dataset_detail'),
    path('datasets/<uuid:dataset_id>/rate/', views.dataset_rate, name='dataset_rate'),
    path('datasets/<uuid:dataset_id>/questions/', views.dataset_questions, name='dataset_questions'),
    
    # Query Execution
    path('playground/', views.QueryPlaygroundView.as_view(), name='query_playground'),
    path('api/execute/playground/', views.execute_playground_query, name='execute_playground_query'),
    path('api/execute/batch/', views.execute_batch_queries, name='execute_batch_queries'),
    path('api/visualize/plan/', views.visualize_query_plan, name='visualize_query_plan'),
    
    # Question Execution
    path('questions/<uuid:pk>/', views.QuestionDetailView.as_view(), name='question_detail'),
    path('questions/<uuid:question_id>/execute/', views.execute_query, name='execute_query'),
    path('questions/<uuid:question_id>/hint/', views.get_hint, name='get_hint'),
    path('questions/<uuid:question_id>/solution/', views.view_solution, name='view_solution'),
    
    # Batch Processing
    path('batch-processing/', views.BatchProcessingView.as_view(), name='batch_processing'),
    path('api/datasets/batch-process/', views.process_csv_folder, name='batch_process'),
    
    # AI Features
    path('api/ai/generate-practice/', views.generate_ai_practice, name='generate_ai_practice'),
    path('api/ai/explain-query/', views.explain_query, name='explain_query'),
    path('api/ai/feedback/', views.get_ai_feedback, name='ai_feedback'),
    
    # User Progress
    path('profile/', views.UserProfileView.as_view(), name='profile'),
    path('progress/', views.UserProgressView.as_view(), name='progress'),
    path('leaderboard/', views.LeaderboardView.as_view(), name='leaderboard'),
    
    # Learning Paths
    path('learning-paths/', views.LearningPathListView.as_view(), name='learningpath_list'),
    path('learning-paths/<uuid:pk>/', views.LearningPathDetailView.as_view(), name='learningpath_detail'),
    path('learning-paths/<uuid:path_id>/start/', views.start_learning_path, name='start_learning_path'),
    
    # Authentication
    path('accounts/', include('django.contrib.auth.urls')),
    path('signup/', views.SignUpView.as_view(), name='signup'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
]