# views.py - Enhanced SQL Learning Platform Views
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView
from django.http import JsonResponse, HttpResponse, Http404, HttpResponseBadRequest
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.db.models import Q, Count, Sum, Avg, F
from django.db import transaction
from django.utils import timezone
from django.conf import settings
from django.urls import reverse_lazy
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import json
import os
import pandas as pd
import uuid
import time
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
    check_achievements, update_user_progress, create_dataset_from_csv,
    check_query_correctness, get_expected_columns, generate_ai_feedback
)
from .ai_utils import TechySQLAI

# ===============================
# Dataset Management Views
# ===============================

class DatasetUploadView(LoginRequiredMixin, CreateView):
    """Enhanced dataset upload with automatic processing"""
    model = Dataset
    form_class = DatasetUploadForm
    template_name = 'sql_platform/dataset_upload.html'
    success_url = reverse_lazy('dataset_list')

    def form_valid(self, form):
        try:
            with transaction.atomic():
                form.instance.created_by = self.request.user
                form.instance.slug = self.generate_slug(form.instance.name)
                
                # Save the dataset first to get an ID
                self.object = form.save()
                
                # Process CSV and create database
                csv_file = form.cleaned_data['csv_file']
                result = self.process_dataset(csv_file, self.object)
                
                if not result['success']:
                    raise Exception(result['error'])
                
                messages.success(self.request, 
                    f"Dataset '{self.object.name}' created successfully with {result['records_loaded']} records!")
                
                # Generate sample questions if requested
                if form.cleaned_data.get('generate_sample_questions'):
                    self.generate_initial_questions(self.object)
                
                return redirect(self.get_success_url())
        
        except Exception as e:
            messages.error(self.request, f"Dataset creation failed: {str(e)}")
            return self.form_invalid(form)

    def process_dataset(self, csv_file, dataset):
        """Process uploaded CSV and create database"""
        try:
            # Save CSV to media folder
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'datasets/csv'))
            filename = fs.save(f"{dataset.id}.csv", csv_file)
            csv_path = fs.path(filename)
            
            # Create dataset in database
            result = create_dataset_from_csv(
                csv_path=csv_path,
                dataset_name=dataset.name,
                industry_name=dataset.industry.name,
                difficulty=dataset.difficulty,
                created_by=dataset.created_by
            )
            
            # Update dataset with generated schema
            if result['success']:
                dataset.schema = result['schema']
                dataset.sample_data = result['sample_data']
                dataset.csv_file.name = f'datasets/csv/{filename}'
                dataset.save()
            
            return result
        
        except Exception as e:
            # Clean up if anything fails
            if 'dataset' in locals():
                dataset.delete()
            if 'csv_path' in locals() and os.path.exists(csv_path):
                os.remove(csv_path)
            raise

    def generate_initial_questions(self, dataset):
        """Generate initial set of questions for the dataset"""
        try:
            # Generate 3 basic questions automatically
            ai = TechySQLAI()
            questions = ai.generate_questions_from_schema(
                dataset.schema,
                count=3,
                difficulty_levels=['EASY']
            )
            
            for q in questions:
                Question.objects.create(
                    dataset=dataset,
                    title=q['title'],
                    description=q['description'],
                    solution_query=q['solution'],
                    difficulty=q['difficulty'],
                    category=QuestionCategory.objects.filter(name__iexact=q['category']).first(),
                    created_by=dataset.created_by,
                    is_published=True
                )
            
            messages.info(self.request, "Generated 3 sample questions for this dataset")
        
        except Exception as e:
            messages.warning(self.request, f"Couldn't generate sample questions: {str(e)}")

    def generate_slug(self, name):
        """Generate unique slug for dataset"""
        base_slug = slugify(name)
        unique_slug = base_slug
        num = 1
        
        while Dataset.objects.filter(slug=unique_slug).exists():
            unique_slug = f"{base_slug}-{num}"
            num += 1
        
        return unique_slug

class DatasetListView(ListView):
    """Enhanced dataset listing with filters and search"""
    model = Dataset
    template_name = 'sql_platform/dataset_list.html'
    context_object_name = 'datasets'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = Dataset.objects.filter(is_published=True).select_related(
            'industry', 'created_by'
        ).prefetch_related('questions').annotate(
            question_count=Count('questions', filter=Q(questions__is_published=True))
        )
        
        # Apply filters from GET parameters
        filters = {
            'industry': self.request.GET.get('industry'),
            'difficulty': self.request.GET.get('difficulty'),
            'search': self.request.GET.get('q'),
            'sort': self.request.GET.get('sort', 'popular')
        }
        
        if filters['industry']:
            queryset = queryset.filter(industry__slug=filters['industry'])
        
        if filters['difficulty']:
            queryset = queryset.filter(difficulty=filters['difficulty'])
        
        if filters['search']:
            queryset = queryset.filter(
                Q(name__icontains=filters['search']) |
                Q(description__icontains=filters['search']) |
                Q(tags__icontains=filters['search'])
            )
        
        # Apply sorting
        if filters['sort'] == 'newest':
            queryset = queryset.order_by('-created_at')
        elif filters['sort'] == 'difficulty':
            queryset = queryset.order_by('difficulty')
        elif filters['sort'] == 'rating':
            queryset = queryset.order_by('-avg_rating')
        else:  # Default: popular
            queryset = queryset.order_by('-total_attempts')
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['industries'] = Industry.objects.annotate(
            dataset_count=Count('datasets', filter=Q(datasets__is_published=True))
        ).filter(dataset_count__gt=0)
        context['current_filters'] = {
            'industry': self.request.GET.get('industry'),
            'difficulty': self.request.GET.get('difficulty'),
            'search': self.request.GET.get('q'),
            'sort': self.request.GET.get('sort', 'popular')
        }
        return context

# ===============================
# Question & Query Execution Views
# ===============================

class QuestionDetailView(LoginRequiredMixin, DetailView):
    """Enhanced question interface with SQL editor"""
    model = Question
    template_name = 'sql_platform/question_detail.html'
    context_object_name = 'question'
    
    def get_queryset(self):
        return Question.objects.filter(is_published=True).select_related(
            'dataset', 'category', 'created_by'
        ).prefetch_related('dependencies')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        question = self.object
        user = self.request.user
        
        # User's previous attempts
        context['user_attempts'] = UserAttempt.objects.filter(
            user=user, question=question
        ).order_by('-attempt_time')[:5]
        
        # Best attempt if any
        context['best_attempt'] = UserAttempt.objects.filter(
            user=user, question=question, is_correct=True
        ).order_by('execution_time_ms').first()
        
        # Question navigation
        dataset_questions = question.dataset.questions.filter(
            is_published=True
        ).order_by('order')
        
        question_ids = list(dataset_questions.values_list('id', flat=True))
        current_index = question_ids.index(question.id)
        
        context['prev_question'] = dataset_questions[current_index - 1] if current_index > 0 else None
        context['next_question'] = dataset_questions[current_index + 1] if current_index < len(question_ids) - 1 else None
        
        # Initialize editor context
        context['editor_config'] = {
            'starter_code': question.starter_code,
            'theme': self.request.user.sql_profile.editor_theme or 'vs-light',
            'font_size': self.request.user.sql_profile.editor_font_size or 14
        }
        
        return context

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def execute_query(request, question_id):
    """Execute and validate user's SQL query"""
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
            
            # Update question statistics
            question.update_statistics()
            
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
            
            attempt.ai_feedback = ai_feedback
            attempt.save()
            
            return JsonResponse({
                'success': True,
                'correct': False,
                'data': result['data'],
                'columns': result['columns'],
                'execution_time': result['execution_time'],
                'ai_feedback': ai_feedback,
                'expected_columns': get_expected_columns(question),
                'differences': correctness.get('differences', []),
                'message': 'Query executed but results are incorrect'
            })
    
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data',
            'type': 'invalid_json'
        }, status=400)
    
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred',
            'type': 'server_error'
        }, status=500)

@login_required
@require_http_methods(["POST"])
def get_hint(request, question_id):
    """Get progressive hints for a question"""
    try:
        question = get_object_or_404(Question, id=question_id, is_published=True)
        hint_level = int(request.POST.get('level', 1))
        
        # Track hint usage in latest attempt
        attempt = UserAttempt.objects.filter(
            user=request.user,
            question=question
        ).order_by('-attempt_time').first()
        
        if attempt:
            attempt.hints_used = max(attempt.hints_used, hint_level)
            attempt.save()
        
        hints = {
            1: question.hint_level_1,
            2: question.hint_level_2,
            3: question.hint_level_3
        }
        
        hint_text = hints.get(hint_level, 'No more hints available')
        
        return JsonResponse({
            'success': True,
            'hint': hint_text,
            'level': hint_level,
            'has_next': hint_level < 3 and bool(hints.get(hint_level + 1))
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

# ===============================
# AI-Powered Features
# ===============================

@login_required
@require_http_methods(["POST"])
def generate_practice_questions(request):
    """Generate personalized practice questions based on user weaknesses"""
    try:
        user = request.user
        profile = user.sql_profile
        
        # Analyze user's weak areas
        weak_categories = [
            cat for cat, stats in profile.get_skill_mastery().items()
            if stats['percentage'] < 70
        ]
        
        # Get datasets matching user's preferred difficulty
        datasets = Dataset.objects.filter(
            difficulty__in=self.get_difficulty_range(profile.preferred_difficulty),
            is_published=True
        )
        
        # Generate questions using AI
        ai = TechySQLAI()
        questions = []
        
        for dataset in datasets[:3]:  # Limit to 3 datasets
            if weak_categories:
                questions.extend(ai.generate_questions_from_schema(
                    dataset.schema,
                    count=2,
                    difficulty_levels=['MEDIUM'],
                    categories=weak_categories
                ))
            else:
                questions.extend(ai.generate_questions_from_schema(
                    dataset.schema,
                    count=2,
                    difficulty_levels=['MEDIUM']
                ))
        
        # Save generated questions as unpublished
        created_questions = []
        for q in questions[:5]:  # Limit to 5 questions
            question = Question.objects.create(
                dataset=Dataset.objects.get(name=q['dataset']),
                title=q['title'],
                description=q['description'],
                solution_query=q['solution'],
                difficulty=q['difficulty'],
                category=QuestionCategory.objects.filter(name__iexact=q['category']).first(),
                created_by=user,
                is_published=False,
                is_ai_generated=True
            )
            created_questions.append({
                'id': question.id,
                'title': question.title,
                'dataset': question.dataset.name
            })
        
        return JsonResponse({
            'success': True,
            'questions': created_questions,
            'weak_categories': weak_categories
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

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
        
        ai = TechySQLAI()
        explanation = ai.explain_query(query)
        
        return JsonResponse({
            'success': True,
            'explanation': explanation
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# ===============================
# User Progress & Analytics
# ===============================

class DashboardView(LoginRequiredMixin, TemplateView):
    """Personalized user dashboard with analytics"""
    template_name = 'sql_platform/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        profile = user.sql_profile
        
        # Recent activity
        context['recent_attempts'] = UserAttempt.objects.filter(
            user=user
        ).select_related('question', 'question__dataset').order_by('-attempt_time')[:10]
        
        # Skill mastery
        context['skill_mastery'] = profile.get_skill_mastery()
        
        # Achievements
        context['recent_achievements'] = UserAchievement.objects.filter(
            user=user,
            is_completed=True
        ).select_related('achievement').order_by('-unlocked_at')[:5]
        
        # Progress charts
        thirty_days_ago = timezone.now() - timedelta(days=30)
        context['progress_data'] = self.get_progress_data(user, thirty_days_ago)
        
        # Recommended content
        context['recommendations'] = self.get_recommendations(profile)
        
        return context
    
    def get_progress_data(self, user, since_date):
        """Get user progress data for charts"""
        attempts = UserAttempt.objects.filter(
            user=user,
            attempt_time__gte=since_date
        ).extra({
            'date': "date(attempt_time)"
        }).values('date').annotate(
            total=Count('id'),
            correct=Count('id', filter=Q(is_correct=True))
        ).order_by('date')
        
        return {
            'dates': [item['date'] for item in attempts],
            'total': [item['total'] for item in attempts],
            'correct': [item['correct'] for item in attempts]
        }
    
    def get_recommendations(self, profile):
        """Get personalized recommendations"""
        # Datasets matching user's skill level
        datasets = Dataset.objects.filter(
            difficulty__in=self.get_difficulty_range(profile.preferred_difficulty),
            is_published=True
        ).exclude(
            id__in=profile.datasets_completed.values_list('id', flat=True)
        ).order_by('-avg_rating')[:5]
        
        # Questions targeting weak areas
        weak_categories = [
            cat for cat, stats in profile.get_skill_mastery().items()
            if stats['percentage'] < 70
        ]
        
        questions = Question.objects.filter(
            category__name__in=weak_categories,
            difficulty__in=self.get_difficulty_range(profile.preferred_difficulty),
            is_published=True
        ).exclude(
            id__in=UserAttempt.objects.filter(
                user=profile.user,
                is_correct=True
            ).values_list('question_id', flat=True)
        ).order_by('?')[:5]  # Random 5 questions
        
        return {
            'datasets': datasets,
            'questions': questions,
            'weak_categories': weak_categories
        }
    
    def get_difficulty_range(self, preferred_level):
        """Get appropriate difficulty levels based on user preference"""
        levels = {
            'NOVICE': ['BEGINNER'],
            'BEGINNER': ['BEGINNER', 'INTERMEDIATE'],
            'INTERMEDIATE': ['INTERMEDIATE'],
            'ADVANCED': ['INTERMEDIATE', 'ADVANCED'],
            'EXPERT': ['ADVANCED', 'EXPERT']
        }
        return levels.get(preferred_level, ['BEGINNER', 'INTERMEDIATE'])

# ===============================
# Dataset Management Utilities
# ===============================

@login_required
@require_http_methods(["POST"])
def process_csv_folder(request):
    """
    Process a folder of CSV files to create datasets
    (For admin/batch processing)
    """
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        csv_folder = os.path.join(settings.MEDIA_ROOT, 'datasets/batch_upload')
        processed = []
        errors = []
        
        for filename in os.listdir(csv_folder):
            if filename.endswith('.csv'):
                try:
                    dataset_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                    csv_path = os.path.join(csv_folder, filename)
                    
                    # Create dataset - using default industry and difficulty
                    result = create_dataset_from_csv(
                        csv_path=csv_path,
                        dataset_name=dataset_name,
                        industry_name="General",
                        difficulty="INTERMEDIATE",
                        created_by=request.user
                    )
                    
                    if result['success']:
                        processed.append(dataset_name)
                        os.rename(csv_path, os.path.join(csv_folder, 'processed', filename))
                    else:
                        errors.append(f"{filename}: {result['error']}")
                
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
                    continue
        
        return JsonResponse({
            'success': True,
            'processed': processed,
            'errors': errors
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# ===============================
# Error Handling
# ===============================

def custom_404(request, exception):
    return render(request, 'sql_platform/404.html', status=404)

def custom_500(request):
    return render(request, 'sql_platform/500.html', status=500)

def custom_403(request, exception):
    return render(request, 'sql_platform/403.html', status=403)

def custom_400(request, exception):
    return render(request, 'sql_platform/400.html', status=400)