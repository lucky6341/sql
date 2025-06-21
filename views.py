# views.py - Enhanced SQL Learning Platform Views
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView as AuthLoginView, LogoutView as AuthLogoutView
from django.contrib.auth.forms import UserCreationForm
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView, View
from django.http import JsonResponse, HttpResponse, Http404, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.db.models import Q, Count, Sum, Avg, F
from django.db import transaction, DatabaseError
from django.utils import timezone
from django.conf import settings
from django.urls import reverse_lazy
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.contrib.auth import login
import json
import os
import pandas as pd
import uuid
import time
import logging
from datetime import datetime, timedelta
from .models import (
    Dataset, Question, UserProfile, UserAttempt, Achievement,
    UserAchievement, Industry, QuestionCategory, LearningPath,
    DatasetRating, AIQuestionGeneration
)
from .forms import DatasetUploadForm, QuestionForm, UserProfileForm
from .utils import (
    execute_sql_query, validate_sql_query, generate_ai_questions,
    calculate_query_performance, analyze_query_complexity,
    check_achievements, create_dataset_from_csv,
    check_query_correctness, generate_ai_feedback
)

logger = logging.getLogger(__name__)

# ===============================
# Home and Authentication Views
# ===============================

class HomeView(TemplateView):
    """TechySQL Academy Homepage"""
    template_name = 'sql_code_editor/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Platform statistics
        context['stats'] = {
            'total_users': UserProfile.objects.count(),
            'total_datasets': Dataset.objects.filter(is_published=True).count(),
            'total_questions': Question.objects.filter(is_published=True).count(),
            'total_attempts': UserAttempt.objects.count(),
        }
        
        # Featured datasets
        context['featured_datasets'] = Dataset.objects.filter(
            is_published=True, is_featured=True
        ).order_by('-created_at')[:3]
        
        return context

class SignUpView(CreateView):
    """User registration view"""
    form_class = UserCreationForm
    template_name = 'registration/signup.html'
    success_url = reverse_lazy('sql_code_editor:dashboard')
    
    def form_valid(self, form):
        response = super().form_valid(form)
        # Log the user in after successful registration
        login(self.request, self.object)
        messages.success(self.request, 'Welcome to TechySQL Academy!')
        return response

class LoginView(AuthLoginView):
    """Custom login view"""
    template_name = 'registration/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('sql_code_editor:dashboard')

class LogoutView(AuthLogoutView):
    """Custom logout view"""
    next_page = reverse_lazy('sql_code_editor:sql_home')

# ===============================
# Dataset Management Views
# ===============================

class DatasetListView(ListView):
    """Enhanced dataset listing with filters"""
    model = Dataset
    template_name = 'sql_code_editor/dataset_list.html'
    context_object_name = 'datasets'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = Dataset.objects.filter(is_published=True).select_related(
            'industry', 'created_by'
        ).annotate(
            question_count=Count('questions', filter=Q(questions__is_published=True))
        )
        
        # Apply filters
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(description__icontains=search) |
                Q(tags__icontains=search)
            )
        
        industry = self.request.GET.get('industry')
        if industry:
            queryset = queryset.filter(industry__slug=industry)
        
        difficulty = self.request.GET.get('difficulty')
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
        
        # Sorting
        sort = self.request.GET.get('sort', '-created_at')
        if sort in ['name', '-name', 'difficulty', '-difficulty', '-avg_rating', '-total_attempts']:
            queryset = queryset.order_by(sort)
        else:
            queryset = queryset.order_by('-created_at')
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['industries'] = Industry.objects.all()
        return context

class DatasetDetailView(DetailView):
    """Detailed dataset view with questions and analytics"""
    model = Dataset
    template_name = 'sql_code_editor/dataset_detail.html'
    context_object_name = 'dataset'
    
    def get_queryset(self):
        return Dataset.objects.filter(is_published=True).select_related(
            'industry', 'created_by'
        ).prefetch_related('questions', 'ratings')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset = self.object
        
        # Questions for this dataset
        context['questions'] = dataset.questions.filter(
            is_published=True
        ).order_by('order', 'difficulty')
        
        # User progress if authenticated
        if self.request.user.is_authenticated:
            solved_questions = UserAttempt.objects.filter(
                user=self.request.user,
                question__dataset=dataset,
                is_correct=True
            ).values_list('question_id', flat=True)
            context['solved_questions'] = list(solved_questions)
            context['progress_percentage'] = (
                len(solved_questions) / context['questions'].count() * 100
                if context['questions'].count() > 0 else 0
            )
        
        # Recent ratings
        context['recent_ratings'] = dataset.ratings.select_related(
            'user'
        ).order_by('-created_at')[:5]
        
        return context

class DatasetUploadView(LoginRequiredMixin, CreateView):
    """Enhanced dataset upload with automatic processing"""
    model = Dataset
    form_class = DatasetUploadForm
    template_name = 'sql_code_editor/dataset_upload.html'
    success_url = reverse_lazy('sql_code_editor:dataset_list')
    
    def form_valid(self, form):
        try:
            with transaction.atomic():
                form.instance.created_by = self.request.user
                form.instance.slug = self.generate_slug(form.instance.name)
                
                # Save the dataset first to get an ID
                self.object = form.save()
                
                # Process CSV and create database
                csv_file = form.cleaned_data['csv_file']
                result = create_dataset_from_csv(
                    csv_file=csv_file,
                    dataset_name=self.object.name,
                    industry_name=self.object.industry.name,
                    difficulty=self.object.difficulty,
                    created_by=self.request.user
                )
                
                if result['success']:
                    self.object.schema = result['schema']
                    self.object.sample_data = result['sample_data']
                    self.object.save()
                    
                    messages.success(
                        self.request, 
                        f"Dataset '{self.object.name}' uploaded successfully!"
                    )
                    
                    # Generate sample questions if requested
                    if form.cleaned_data.get('generate_sample_questions'):
                        # This would be done asynchronously in production
                        pass
                    
                    return redirect(self.get_success_url())
                else:
                    self.object.delete()
                    messages.error(self.request, f"Upload failed: {result['error']}")
                    return self.form_invalid(form)
        
        except Exception as e:
            logger.error(f"Dataset creation failed: {str(e)}", exc_info=True)
            messages.error(self.request, f"Dataset creation failed: {str(e)}")
            return self.form_invalid(form)
    
    def generate_slug(self, name):
        """Generate unique slug for dataset"""
        from django.utils.text import slugify
        base_slug = slugify(name)
        slug = base_slug
        counter = 1
        
        while Dataset.objects.filter(slug=slug).exists():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug

# ===============================
# Query Execution Views
# ===============================

class QueryPlaygroundView(LoginRequiredMixin, TemplateView):
    """SQL Playground for freeform query execution"""
    template_name = 'sql_code_editor/query_playground.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['datasets'] = Dataset.objects.filter(is_published=True).order_by('name')
        context['editor_config'] = {
            'theme': getattr(self.request.user.sql_profile, 'editor_theme', 'vs-dark'),
            'font_size': getattr(self.request.user.sql_profile, 'editor_font_size', 14)
        }
        return context

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def execute_playground_query(request):
    """Execute freeform SQL query in playground mode"""
    try:
        data = json.loads(request.body)
        user_query = data.get('query', '').strip()
        dataset_id = data.get('dataset_id')
        
        if not user_query:
            return JsonResponse({
                'success': False,
                'error': 'Query cannot be empty',
                'type': 'empty_query'
            }, status=400)
        
        if not dataset_id:
            return JsonResponse({
                'success': False,
                'error': 'Dataset not selected',
                'type': 'missing_dataset'
            }, status=400)
        
        # Execute query against dataset
        result = execute_sql_query(user_query, dataset_id, timeout=120)
        
        if result['success']:
            # Add performance metrics
            if result.get('data'):
                result['performance'] = calculate_query_performance(
                    user_query,
                    result['data'],
                    result['execution_time']
                )
            return JsonResponse(result)
        else:
            return JsonResponse(result, status=400)
    
    except Exception as e:
        logger.error(f"Playground query execution failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred',
            'type': 'server_error'
        }, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def execute_query(request, question_id):
    """Execute and validate user's SQL query with enhanced feedback"""
    try:
        question = get_object_or_404(Question, id=question_id, is_published=True)
        data = json.loads(request.body)
        user_query = data.get('query', '').strip()
        start_time = time.time()
        
        if not user_query:
            return JsonResponse({
                'success': False,
                'error': 'Query cannot be empty',
                'type': 'empty_query'
            }, status=400)
        
        # Validate SQL query syntax
        validation = validate_sql_query(user_query)
        if not validation['valid']:
            return JsonResponse({
                'success': False,
                'error': validation['error'],
                'type': 'invalid_sql'
            }, status=400)
        
        # Execute query against dataset
        result = execute_sql_query(
            user_query,
            question.dataset.id,
            timeout=question.time_limit_minutes * 60 if question.time_limit_minutes else 30
        )
        
        if not result['success']:
            # Log failed attempt
            UserAttempt.objects.create(
                user=request.user,
                question=question,
                user_query=user_query,
                status='ERROR',
                is_correct=False,
                error_message=result['error'],
                attempts_count=UserAttempt.objects.filter(
                    user=request.user,
                    question=question
                ).count() + 1,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            
            return JsonResponse({
                'success': False,
                'error': result['error'],
                'type': result.get('type', 'execution_error')
            }, status=400)
        
        # Check correctness against solution
        correctness = check_query_correctness(
            user_query,
            question.solution_query,
            question.dataset.id
        )
        
        # Calculate performance metrics
        performance = calculate_query_performance(
            user_query,
            result['data'],
            result['execution_time']
        )
        
        # Create attempt record
        attempt = UserAttempt.objects.create(
            user=request.user,
            question=question,
            user_query=user_query,
            status='CORRECT' if correctness['correct'] else 'INCORRECT',
            is_correct=correctness['correct'],
            execution_time_ms=result['execution_time'],
            time_taken=int(time.time() - start_time),
            attempts_count=UserAttempt.objects.filter(
                user=request.user,
                question=question
            ).count() + 1,
            result_data=result['data'],
            performance_score=performance['performance_score'],
            query_complexity=analyze_query_complexity(user_query)
        )
        
        # Handle correct answers
        if correctness['correct']:
            points = attempt.calculate_points()
            attempt.points_earned = points
            attempt.save()
            
            # Update user profile
            profile = request.user.sql_profile
            profile.total_points += points
            profile.experience_points += points
            
            # Only count as new solved question if first correct attempt
            if not UserAttempt.objects.filter(
                user=request.user,
                question=question,
                is_correct=True,
                attempt_time__lt=attempt.attempt_time
            ).exists():
                profile.questions_solved += 1
            
            profile.save()
            
            # Check for level up
            level_up = profile.update_level()
            
            # Check achievements
            new_achievements = check_achievements(request.user, attempt)
            
            return JsonResponse({
                'success': True,
                'correct': True,
                'data': result['data'],
                'columns': result['columns'],
                'execution_time': result['execution_time'],
                'points_earned': points,
                'level_up': level_up,
                'achievements': new_achievements,
                'performance': performance,
                'message': 'Correct solution!'
            })
        
        else:
            # Generate AI feedback for incorrect answers
            ai_feedback = generate_ai_feedback(
                user_query,
                question.solution_query,
                result,
                execute_sql_query(question.solution_query, question.dataset.id)
            )
            
            attempt.ai_feedback = json.dumps(ai_feedback)
            attempt.save()
            
            return JsonResponse({
                'success': True,
                'correct': False,
                'data': result['data'],
                'columns': result['columns'],
                'execution_time': result['execution_time'],
                'ai_feedback': ai_feedback,
                'differences': correctness.get('differences', []),
                'message': 'Query executed but results are incorrect'
            })
    
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred',
            'type': 'server_error'
        }, status=500)

# ===============================
# User Progress Views
# ===============================

class DashboardView(LoginRequiredMixin, TemplateView):
    """Personalized user dashboard"""
    template_name = 'sql_code_editor/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        profile = user.sql_profile
        
        # Recent attempts
        context['recent_attempts'] = UserAttempt.objects.filter(
            user=user
        ).select_related('question', 'question__dataset').order_by('-attempt_time')[:10]
        
        # Recent achievements
        context['recent_achievements'] = UserAchievement.objects.filter(
            user=user,
            is_completed=True
        ).select_related('achievement').order_by('-unlocked_at')[:5]
        
        # Recommendations
        context['recommendations'] = self.get_recommendations(user)
        
        return context
    
    def get_recommendations(self, user):
        """Get personalized recommendations"""
        # Get user's weak areas
        skill_mastery = user.sql_profile.get_skill_mastery()
        weak_categories = [
            cat for cat, stats in skill_mastery.items()
            if stats['percentage'] < 70
        ]
        
        # Recommend questions in weak areas
        recommended_questions = Question.objects.filter(
            category__name__in=weak_categories[:3],
            is_published=True
        ).exclude(
            id__in=UserAttempt.objects.filter(
                user=user, is_correct=True
            ).values_list('question_id', flat=True)
        )[:5]
        
        # Recommend learning paths
        learning_paths = LearningPath.objects.filter(
            difficulty=user.sql_profile.preferred_difficulty,
            is_published=True
        )[:3]
        
        return {
            'questions': recommended_questions,
            'learning_paths': learning_paths
        }

class UserProfileView(LoginRequiredMixin, UpdateView):
    """User profile management"""
    model = UserProfile
    form_class = UserProfileForm
    template_name = 'sql_code_editor/profile.html'
    success_url = reverse_lazy('sql_code_editor:profile')
    
    def get_object(self):
        return self.request.user.sql_profile

class LeaderboardView(ListView):
    """Global leaderboard"""
    model = UserProfile
    template_name = 'sql_code_editor/leaderboard.html'
    context_object_name = 'profiles'
    paginate_by = 50
    
    def get_queryset(self):
        return UserProfile.objects.select_related('user').order_by(
            '-total_points', '-questions_solved'
        )

# ===============================
# Learning Path Views
# ===============================

class LearningPathListView(ListView):
    """Learning paths listing"""
    model = LearningPath
    template_name = 'sql_code_editor/learning_paths.html'
    context_object_name = 'learning_paths'
    
    def get_queryset(self):
        return LearningPath.objects.filter(is_published=True).prefetch_related(
            'datasets'
        ).order_by('order', 'name')

class LearningPathDetailView(DetailView):
    """Learning path detail view"""
    model = LearningPath
    template_name = 'sql_code_editor/learning_path_detail.html'
    context_object_name = 'learning_path'
    
    def get_queryset(self):
        return LearningPath.objects.filter(is_published=True).prefetch_related(
            'datasets__dataset'
        )

# ===============================
# AI Features
# ===============================

@login_required
@require_http_methods(["POST"])
def explain_query(request):
    """Explain SQL query in natural language using AI"""
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({
                'success': False,
                'error': 'Query cannot be empty'
            }, status=400)
        
        # Mock AI explanation (replace with actual AI service)
        explanation = f"This SQL query performs the following operations:\n\n"
        explanation += "1. Selects data from tables\n"
        explanation += "2. Applies filtering conditions\n"
        explanation += "3. Returns the result set\n\n"
        explanation += "The query appears to be well-structured and should execute efficiently."
        
        return JsonResponse({
            'success': True,
            'explanation': explanation
        })
    
    except Exception as e:
        logger.error(f"Query explanation failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# ===============================
# Utility Views
# ===============================

@login_required
def dataset_rate(request, dataset_id):
    """Rate a dataset"""
    if request.method == 'POST':
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id)
            rating = int(request.POST.get('rating', 0))
            review = request.POST.get('review', '')
            
            if not (1 <= rating <= 5):
                messages.error(request, 'Rating must be between 1 and 5')
                return redirect('sql_code_editor:dataset_detail', pk=dataset_id)
            
            rating_obj, created = DatasetRating.objects.update_or_create(
                user=request.user,
                dataset=dataset,
                defaults={'rating': rating, 'review': review}
            )
            
            # Update dataset statistics
            dataset.update_statistics()
            
            if created:
                messages.success(request, 'Thank you for rating this dataset!')
            else:
                messages.success(request, 'Your rating has been updated!')
            
        except Exception as e:
            messages.error(request, f'Error saving rating: {str(e)}')
    
    return redirect('sql_code_editor:dataset_detail', pk=dataset_id)

@login_required
def start_learning_path(request, path_id):
    """Start a learning path"""
    learning_path = get_object_or_404(LearningPath, id=path_id, is_published=True)
    
    # Mark as started (you might want to create a UserLearningPath model)
    messages.success(
        request, 
        f'You have started the "{learning_path.name}" learning path!'
    )
    
    return redirect('sql_code_editor:learningpath_detail', pk=path_id)

# ===============================
# API ViewSets (from views/api.py)
# ===============================

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import (
    DatasetSerializer, QuestionSerializer, UserAttemptSerializer,
    AchievementSerializer, LearningPathSerializer
)

class DatasetViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Dataset.objects.filter(is_published=True)
    serializer_class = DatasetSerializer

class QuestionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Question.objects.filter(is_published=True)
    serializer_class = QuestionSerializer

class UserAttemptViewSet(viewsets.ModelViewSet):
    serializer_class = UserAttemptSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserAttempt.objects.filter(user=self.request.user)

class AchievementViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Achievement.objects.filter(is_active=True)
    serializer_class = AchievementSerializer

class LearningPathViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = LearningPath.objects.filter(is_published=True)
    serializer_class = LearningPathSerializer

# ===============================
# Batch Processing (Placeholder)
# ===============================

class BatchProcessingView(LoginRequiredMixin, TemplateView):
    """Batch processing interface"""
    template_name = 'sql_code_editor/batch_processing.html'
    
    def get(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return HttpResponseForbidden("Only staff members can access batch processing")
        return super().get(request, *args, **kwargs)

@login_required
@require_http_methods(["POST"])
def process_csv_folder(request):
    """Process CSV folder (admin only)"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    # Placeholder implementation
    return JsonResponse({
        'success': True,
        'message': 'Batch processing completed',
        'processed': [],
        'errors': []
    })

# Additional utility functions
@login_required
def get_hint(request, question_id):
    """Get hint for a question"""
    question = get_object_or_404(Question, id=question_id, is_published=True)
    hint_level = int(request.GET.get('level', 1))
    
    hints = {
        1: question.hint_level_1,
        2: question.hint_level_2,
        3: question.hint_level_3
    }
    
    hint = hints.get(hint_level, '')
    
    return JsonResponse({
        'success': True,
        'hint': hint,
        'level': hint_level
    })

@login_required
def view_solution(request, question_id):
    """View solution for a question (if user has permission)"""
    question = get_object_or_404(Question, id=question_id, is_published=True)
    
    # Check if user has solved the question or has permission
    has_solved = UserAttempt.objects.filter(
        user=request.user,
        question=question,
        is_correct=True
    ).exists()
    
    if not has_solved and not request.user.is_staff:
        return JsonResponse({
            'success': False,
            'error': 'You must solve the question first to view the solution'
        }, status=403)
    
    return JsonResponse({
        'success': True,
        'solution': question.solution_query,
        'explanation': question.explanation
    })

def dataset_questions(request, dataset_id):
    """Get questions for a dataset"""
    dataset = get_object_or_404(Dataset, id=dataset_id, is_published=True)
    questions = dataset.questions.filter(is_published=True).order_by('order')
    
    questions_data = []
    for question in questions:
        questions_data.append({
            'id': str(question.id),
            'title': question.title,
            'difficulty': question.difficulty,
            'points': question.points,
            'success_rate': question.success_rate
        })
    
    return JsonResponse({
        'success': True,
        'questions': questions_data
    })

# Placeholder AI functions
@login_required
@require_http_methods(["POST"])
def generate_ai_practice(request):
    """Generate AI practice questions"""
    return JsonResponse({
        'success': True,
        'questions': [],
        'message': 'AI practice generation will be implemented with API keys'
    })

@login_required
@require_http_methods(["POST"])
def get_ai_feedback(request):
    """Get AI feedback on query"""
    return JsonResponse({
        'success': True,
        'feedback': 'AI feedback will be implemented with API keys'
    })

@login_required
@require_http_methods(["POST"])
def execute_batch_queries(request):
    """Execute multiple queries in batch"""
    return JsonResponse({
        'success': True,
        'results': [],
        'message': 'Batch execution will be implemented'
    })

@login_required
@require_http_methods(["POST"])
def visualize_query_plan(request):
    """Visualize query execution plan"""
    return JsonResponse({
        'success': True,
        'plan': {},
        'message': 'Query plan visualization will be implemented'
    })