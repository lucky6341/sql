# views.py - Enhanced SQL Learning Platform Views with New Execution Endpoints
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
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
    DatasetRating, AIQuestionGeneration, BatchProcessingJob
)
from .forms import DatasetUploadForm, QuestionForm, UserProfileForm, BatchProcessingForm
from .utils import (
    execute_sql_query, validate_sql_query, generate_ai_questions,
    calculate_query_performance, analyze_query_complexity,
    check_achievements, update_user_progress, create_dataset_from_csv,
    check_query_correctness, get_expected_columns, generate_ai_feedback,
    normalize_data, slugify, format_sql
)
from .ai_utils import TechySQLAI
from .tasks import process_dataset_task, generate_sample_questions_task

logger = logging.getLogger(__name__)

# ===============================
# Dataset Management Views
# ===============================

class DatasetUploadView(LoginRequiredMixin, CreateView):
    """Enhanced dataset upload with automatic processing and async support"""
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
                
                # Process CSV and create database asynchronously
                csv_file = form.cleaned_data['csv_file']
                self.process_dataset_async(csv_file, self.object)
                
                messages.success(self.request, 
                    f"Dataset '{self.object.name}' is being processed. You'll be notified when ready.")
                
                # Generate sample questions if requested
                if form.cleaned_data.get('generate_sample_questions'):
                    generate_sample_questions_task.delay(self.object.id)
                
                return redirect(self.get_success_url())
        
        except Exception as e:
            logger.error(f"Dataset creation failed: {str(e)}", exc_info=True)
            messages.error(self.request, f"Dataset creation failed: {str(e)}")
            return self.form_invalid(form)

    def process_dataset_async(self, csv_file, dataset):
        """Process uploaded CSV asynchronously using Celery"""
        try:
            # Save CSV to media folder
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'datasets/csv'))
            filename = fs.save(f"{dataset.id}.csv", csv_file)
            csv_path = fs.path(filename)
            
            # Start async processing
            process_dataset_task.delay(str(dataset.id), csv_path)
            
        except Exception as e:
            logger.error(f"Error preparing dataset for async processing: {str(e)}")
            raise

    # ... rest of DatasetUploadView remains the same ...

# ===============================
# Enhanced Query Execution Views
# ===============================

class QueryPlaygroundView(LoginRequiredMixin, TemplateView):
    """SQL Playground for freeform query execution"""
    template_name = 'sql_platform/query_playground.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['datasets'] = Dataset.objects.filter(is_published=True).order_by('name')
        context['editor_config'] = {
            'theme': self.request.user.sql_profile.editor_theme or 'vs-light',
            'font_size': self.request.user.sql_profile.editor_font_size or 14
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
        result = execute_sql_query(
            user_query,
            dataset_id,
            timeout=120  # Longer timeout for playground
        )
        
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
            # Generate detailed error explanation
            error_explanation = generate_error_explanation(result)
            result['error_explanation'] = error_explanation
            return JsonResponse(result, status=400)
    
    except Exception as e:
        logger.error(f"Playground query execution failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred',
            'type': 'server_error'
        }, status=500)

def generate_error_explanation(result):
    """Generate user-friendly error explanations"""
    error_type = result.get('type', '')
    details = result.get('error_details', [])
    
    explanations = {
        'syntax_error': "Your SQL query contains syntax errors.",
        'missing_table': "You're referencing a table that doesn't exist.",
        'missing_column': "You're referencing a column that doesn't exist.",
        'ambiguous_column': "Column name is ambiguous (exists in multiple tables).",
        'validation_error': "Your query contains validation issues.",
        'execution_error': "An error occurred while executing your query."
    }
    
    base_message = explanations.get(error_type, "An error occurred with your query.")
    detail_messages = []
    
    for detail in details:
        if detail['type'] == 'forbidden_keyword':
            detail_messages.append(
                f"Keyword '{detail['keyword']}' is not allowed: {detail['message']}"
            )
        elif detail['type'] == 'syntax_error' and 'problem_area' in detail:
            detail_messages.append(
                f"Syntax issue near: '{detail['problem_area']}'. {detail.get('suggestion', '')}"
            )
        elif detail['type'] == 'missing_column':
            detail_messages.append(
                f"Column '{detail['missing_column']}' doesn't exist in the dataset."
            )
    
    if detail_messages:
        return base_message + " " + " ".join(detail_messages)
    return base_message

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def execute_batch_queries(request):
    """Execute multiple queries in a batch"""
    try:
        data = json.loads(request.body)
        queries = data.get('queries', [])
        dataset_id = data.get('dataset_id')
        
        if not queries:
            return JsonResponse({
                'success': False,
                'error': 'No queries provided',
                'type': 'empty_batch'
            }, status=400)
        
        if not dataset_id:
            return JsonResponse({
                'success': False,
                'error': 'Dataset not selected',
                'type': 'missing_dataset'
            }, status=400)
        
        results = []
        total_time = 0
        
        for query in queries:
            if not query.strip():
                continue
                
            start_time = time.time()
            result = execute_sql_query(
                query,
                dataset_id,
                timeout=30
            )
            exec_time = time.time() - start_time
            
            # Add performance metrics
            if result['success'] and result.get('data'):
                result['performance'] = calculate_query_performance(
                    query,
                    result['data'],
                    result['execution_time']
                )
            
            # Add additional metadata
            result['query'] = query
            result['execution_time_seconds'] = exec_time
            total_time += exec_time
            
            results.append(result)
        
        return JsonResponse({
            'success': True,
            'results': results,
            'total_queries': len(results),
            'success_count': sum(1 for r in results if r['success']),
            'total_time': total_time
        })
    
    except Exception as e:
        logger.error(f"Batch query execution failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred',
            'type': 'server_error'
        }, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def visualize_query_plan(request):
    """Generate visual query execution plan"""
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
        
        # Generate EXPLAIN plan
        explain_query = f"EXPLAIN QUERY PLAN {user_query}"
        result = execute_sql_query(
            explain_query,
            dataset_id,
            timeout=30
        )
        
        if not result['success']:
            return JsonResponse(result, status=400)
        
        # Parse and format the query plan
        plan = parse_query_plan(result['data'])
        
        return JsonResponse({
            'success': True,
            'plan': plan,
            'raw_plan': result['data']
        })
    
    except Exception as e:
        logger.error(f"Query plan visualization failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'Failed to generate query plan',
            'type': 'visualization_error'
        }, status=500)

def parse_query_plan(plan_data):
    """Convert raw query plan into structured format"""
    structured_plan = []
    for step in plan_data:
        if 'detail' in step:
            # SQLite format
            parts = step['detail'].split('|')
            if len(parts) >= 4:
                structured_plan.append({
                    'id': parts[0].strip(),
                    'parent': parts[1].strip(),
                    'operation': parts[2].strip(),
                    'details': '|'.join(parts[3:]).strip()
                })
        elif 'EXPLAIN' in step:
            # MySQL format
            structured_plan.append({
                'operation': step['EXPLAIN'].split(' ', 1)[0],
                'details': step['EXPLAIN']
            })
    return structured_plan

# ===============================
# Enhanced Question & Query Execution Views
# ===============================

class QuestionDetailView(LoginRequiredMixin, DetailView):
    """Enhanced question interface with SQL editor and visualizations"""
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
        
        # Add visualization context
        context['visualization_types'] = self.get_visualization_types(question)
        
        # Add performance context
        context['performance_metrics'] = self.get_performance_metrics(question)
        
        # ... rest of existing context ...
        
        return context
    
    def get_visualization_types(self, question):
        """Determine supported visualization types based on result structure"""
        if not question.expected_output or not question.expected_output.get('columns'):
            return []
        
        numeric_cols = 0
        for col in question.expected_output['columns']:
            # Simple heuristic to detect numeric columns
            if any(t in col.lower() for t in ['id', 'count', 'total', 'sum', 'avg', 'price', 'amount']):
                numeric_cols += 1
        
        visualizations = ['table']
        if numeric_cols >= 1:
            visualizations.append('bar')
            visualizations.append('pie')
        if numeric_cols >= 2:
            visualizations.append('scatter')
            visualizations.append('line')
        
        return visualizations
    
    def get_performance_metrics(self, question):
        """Get performance metrics for this question"""
        metrics = {
            'success_rate': question.success_rate,
            'avg_time': question.avg_time_seconds,
            'avg_attempts': question.avg_attempts
        }
        
        # Add percentile ranking if available
        cache_key = f'question_percentile_{question.id}'
        percentile = cache.get(cache_key)
        if percentile is None:
            percentile = self.calculate_percentile(question)
            cache.set(cache_key, percentile, 3600)  # Cache for 1 hour
        metrics['percentile'] = percentile
        
        return metrics
    
    def calculate_percentile(self, question):
        """Calculate performance percentile for this question"""
        all_questions = Question.objects.filter(
            difficulty=question.difficulty,
            is_published=True
        ).exclude(id=question.id).values('id', 'success_rate')
        
        if not all_questions:
            return 0
        
        better_questions = sum(1 for q in all_questions if q['success_rate'] < question.success_rate)
        percentile = int((better_questions / len(all_questions)) * 100)
        return percentile

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
        
        # Validate SQL query syntax with enhanced validation
        validation = validate_sql_query(user_query)
        if not validation['valid']:
            return JsonResponse({
                'success': False,
                'error': validation['error'],
                'type': 'invalid_sql',
                'validation_details': validation.get('details', [])
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
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_details=json.dumps(result.get('error_details', []))
            )
            
            # Generate user-friendly error explanation
            explanation = generate_error_explanation(result)
            
            return JsonResponse({
                'success': False,
                'error': result['error'],
                'error_explanation': explanation,
                'type': result.get('type', 'execution_error'),
                'error_details': result.get('error_details', [])
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
            query_complexity=analyze_query_complexity(user_query),
            correctness_details=json.dumps({
                'match_percentage': correctness.get('match_percentage', 0),
                'differences': correctness.get('differences', [])
            })
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
            
            attempt.ai_feedback = json.dumps(ai_feedback)
            attempt.save()
            
            # Add detailed comparison data
            response_data = {
                'success': True,
                'correct': False,
                'data': result['data'],
                'columns': result['columns'],
                'execution_time': result['execution_time'],
                'ai_feedback': ai_feedback,
                'expected_columns': get_expected_columns(question),
                'differences': correctness.get('differences', []),
                'message': 'Query executed but results are incorrect'
            }
            
            # Add correctness details if available
            if correctness.get('match_percentage') is not None:
                response_data['match_percentage'] = correctness['match_percentage']
                
            return JsonResponse(response_data)
    
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data',
            'type': 'invalid_json'
        }, status=400)
    
    except DatabaseError as e:
        logger.error(f"Database error executing query: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'Database operation failed',
            'type': 'database_error'
        }, status=500)
    
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred',
            'type': 'server_error'
        }, status=500)

# ===============================
# Enhanced Batch Processing Views
# ===============================

class BatchProcessingView(LoginRequiredMixin, CreateView):
    """Batch processing interface for large operations"""
    model = BatchProcessingJob
    form_class = BatchProcessingForm
    template_name = 'sql_platform/batch_processing.html'
    success_url = reverse_lazy('batch_processing')
    
    def form_valid(self, form):
        if not self.request.user.is_staff:
            return HttpResponseForbidden("Only staff members can perform batch processing")
        
        try:
            job = form.save(commit=False)
            job.initiated_by = self.request.user
            job.status = 'PENDING'
            job.save()
            
            # Start processing based on job type
            if job.job_type == 'DATASET_IMPORT':
                self.process_dataset_import(job)
            elif job.job_type == 'QUESTION_GENERATION':
                self.process_question_generation(job)
            
            messages.success(self.request, f"Batch job '{job.name}' started successfully.")
            return redirect(self.success_url)
        
        except Exception as e:
            logger.error(f"Batch job creation failed: {str(e)}", exc_info=True)
            messages.error(self.request, f"Batch job failed: {str(e)}")
            return self.form_invalid(form)
    
    def process_dataset_import(self, job):
        """Process dataset import job"""
        # Implementation would queue async tasks
        job.status = 'PROCESSING'
        job.save()
        # Start async processing (simulated)
        time.sleep(2)
        job.status = 'COMPLETED'
        job.result = {'datasets_processed': 5}
        job.completed_at = timezone.now()
        job.save()
    
    def process_question_generation(self, job):
        """Process question generation job"""
        # Implementation would queue async tasks
        job.status = 'PROCESSING'
        job.save()
        # Start async processing (simulated)
        time.sleep(3)
        job.status = 'COMPLETED'
        job.result = {'questions_generated': 12}
        job.completed_at = timezone.now()
        job.save()

@login_required
@require_http_methods(["POST"])
def process_csv_folder(request):
    """
    Enhanced CSV folder processing with job tracking
    (For admin/batch processing)
    """
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        csv_folder = os.path.join(settings.MEDIA_ROOT, 'datasets/batch_upload')
        processed = []
        errors = []
        
        # Create batch job record
        job = BatchProcessingJob.objects.create(
            name=f"CSV Import {datetime.now().strftime('%Y-%m-%d')}",
            job_type='DATASET_IMPORT',
            initiated_by=request.user,
            status='PROCESSING'
        )
        
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
        
        # Update job status
        job.status = 'COMPLETED'
        job.result = {
            'processed': processed,
            'errors': errors
        }
        job.completed_at = timezone.now()
        job.save()
        
        return JsonResponse({
            'success': True,
            'job_id': str(job.id),
            'processed': processed,
            'errors': errors
        })
    
    except Exception as e:
        logger.error(f"Batch CSV processing failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# ===============================
# Enhanced AI-Powered Features
# ===============================

@login_required
@require_http_methods(["POST"])
def generate_ai_practice(request):
    """Generate personalized practice questions based on user weaknesses with enhanced AI"""
    try:
        user = request.user
        profile = user.sql_profile
        
        # Analyze user's weak areas with more depth
        weak_categories = []
        skill_mastery = profile.get_skill_mastery()
        for cat, stats in skill_mastery.items():
            if stats['percentage'] < 70:
                weak_categories.append({
                    'category': cat,
                    'mastery': stats['percentage'],
                    'total_attempts': stats['total'],
                    'error_rate': 1 - (stats['correct'] / stats['total']) if stats['total'] > 0 else 0
                })
        
        # Sort by weakest first
        weak_categories.sort(key=lambda x: x['mastery'])
        
        # Get datasets matching user's preferred difficulty
        datasets = Dataset.objects.filter(
            difficulty__in=self.get_difficulty_range(profile.preferred_difficulty),
            is_published=True
        ).order_by('?')  # Random order
        
        # Generate questions using AI
        ai = TechySQLAI()
        questions = []
        
        for dataset in datasets[:3]:  # Limit to 3 datasets
            if weak_categories:
                categories = [wc['category'] for wc in weak_categories[:3]]
                questions.extend(ai.generate_questions_from_schema(
                    dataset.schema,
                    count=3,
                    difficulty_levels=['MEDIUM', 'HARD'],
                    categories=categories,
                    focus_areas=[wc['category'] for wc in weak_categories[:2]]
                ))
            else:
                questions.extend(ai.generate_questions_from_schema(
                    dataset.schema,
                    count=3,
                    difficulty_levels=['MEDIUM', 'HARD']
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
                'dataset': question.dataset.name,
                'difficulty': question.difficulty,
                'category': question.category.name
            })
        
        # Update user profile
        profile.last_ai_practice = timezone.now()
        profile.save()
        
        return JsonResponse({
            'success': True,
            'questions': created_questions,
            'weak_categories': [wc['category'] for wc in weak_categories]
        })
    
    except Exception as e:
        logger.error(f"AI practice generation failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@login_required
@require_http_methods(["POST"])
def explain_query(request):
    """Explain SQL query in natural language using AI with enhanced context"""
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        dataset_id = data.get('dataset_id')
        context = data.get('context', '')
        
        if not query:
            return JsonResponse({
                'success': False,
                'error': 'Query cannot be empty'
            }, status=400)
        
        # Get dataset schema if available
        schema = {}
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id)
                schema = dataset.schema
            except Dataset.DoesNotExist:
                pass
        
        ai = TechySQLAI()
        explanation = ai.explain_query(query, schema=schema, context=context)
        
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
# Enhanced User Progress & Analytics
# ===============================

class DashboardView(LoginRequiredMixin, TemplateView):
    """Personalized user dashboard with enhanced analytics"""
    template_name = 'sql_platform/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        profile = user.sql_profile
        
        # Enhanced progress charts
        context['progress_data'] = self.get_enhanced_progress_data(user)
        
        # Skill mastery radar chart data
        context['radar_chart_data'] = self.get_skill_radar_data(profile)
        
        # Learning path recommendations
        context['learning_paths'] = self.get_learning_path_recommendations(profile)
        
        # ... rest of existing context ...
        
        return context
    
    def get_enhanced_progress_data(self, user):
        """Get enhanced progress data for charts"""
        now = timezone.now()
        time_periods = {
            'daily': now - timedelta(days=30),
            'weekly': now - timedelta(weeks=12),
            'monthly': now - timedelta(days=365)
        }
        
        progress_data = {}
        for period, since_date in time_periods.items():
            attempts = UserAttempt.objects.filter(
                user=user,
                attempt_time__gte=since_date
            ).extra({
                'date': "date(attempt_time)"
            }).values('date').annotate(
                total=Count('id'),
                correct=Count('id', filter=Q(is_correct=True))
            ).order_by('date')
            
            progress_data[period] = {
                'dates': [item['date'] for item in attempts],
                'total': [item['total'] for item in attempts],
                'correct': [item['correct'] for item in attempts]
            }
        
        return progress_data
    
    def get_skill_radar_data(self, profile):
        """Format skill mastery data for radar chart"""
        skill_mastery = profile.get_skill_mastery()
        if not skill_mastery:
            return None
        
        # Get top 6 skills
        sorted_skills = sorted(
            skill_mastery.items(), 
            key=lambda x: x[1]['percentage']
        )[:6]
        
        return {
            'labels': [skill[0] for skill in sorted_skills],
            'data': [skill[1]['percentage'] for skill in sorted_skills]
        }
    
    def get_learning_path_recommendations(self, profile):
        """Get recommended learning paths based on skill gaps"""
        weak_skills = [
            skill for skill, stats in profile.get_skill_mastery().items()
            if stats['percentage'] < 70
        ]
        
        if not weak_skills:
            return LearningPath.objects.filter(
                difficulty=profile.preferred_difficulty,
                is_published=True
            ).order_by('?')[:3]
        
        # Find paths that cover weak skills
        return LearningPath.objects.filter(
            datasets__questions__category__name__in=weak_skills,
            difficulty__in=self.get_difficulty_range(profile.preferred_difficulty),
            is_published=True
        ).distinct().annotate(
            relevance=Count('datasets__questions__category', filter=Q(
                datasets__questions__category__name__in=weak_skills
            ))
        ).order_by('-relevance')[:3]

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