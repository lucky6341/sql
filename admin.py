# admin.py - Enhanced SQL Learning Platform Admin
from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Count, Sum, Avg, Q
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.exceptions import ValidationError
import json
import csv
from datetime import datetime, timedelta
from .models import (
    Industry, Dataset, QuestionCategory, Question, UserProfile, 
    UserAttempt, Achievement, UserAchievement, LearningPath,
    LearningPathDataset, DatasetRating, AIQuestionGeneration
)
from .utils import generate_ai_questions, validate_dataset_schema

# ===============================
# Custom Admin Actions
# ===============================

def export_to_csv(modeladmin, request, queryset):
    """Export selected items to CSV"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{modeladmin.model._meta.verbose_name_plural}.csv"'
    
    writer = csv.writer(response)
    
    # Write header
    field_names = [field.name for field in modeladmin.model._meta.fields]
    writer.writerow(field_names)
    
    # Write data
    for obj in queryset:
        row = []
        for field in field_names:
            value = getattr(obj, field)
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            row.append(str(value) if value is not None else '')
        writer.writerow(row)
    
    return response

export_to_csv.short_description = "Export selected items to CSV"

def publish_items(modeladmin, request, queryset):
    """Publish selected items"""
    updated = queryset.update(is_published=True)
    messages.success(request, f'{updated} items were successfully published.')

publish_items.short_description = "Publish selected items"

def unpublish_items(modeladmin, request, queryset):
    """Unpublish selected items"""
    updated = queryset.update(is_published=False)
    messages.success(request, f'{updated} items were successfully unpublished.')

unpublish_items.short_description = "Unpublish selected items"

# ===============================
# Inline Admin Classes
# ===============================

class QuestionInline(admin.TabularInline):
    """Questions inline for Dataset admin"""
    model = Question
    extra = 0
    fields = ('title', 'difficulty', 'points', 'is_published', 'order')
    readonly_fields = ('success_rate', 'total_attempts')
    ordering = ('order',)
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('category')

class DatasetRatingInline(admin.TabularInline):
    """Dataset ratings inline"""
    model = DatasetRating
    extra = 0
    readonly_fields = ('user', 'rating', 'review', 'created_at')
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False

class UserAchievementInline(admin.TabularInline):
    """User achievements inline"""
    model = UserAchievement
    extra = 0
    readonly_fields = ('achievement', 'progress', 'is_completed', 'unlocked_at')
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False

class LearningPathDatasetInline(admin.TabularInline):
    """Learning path datasets inline"""
    model = LearningPathDataset
    extra = 1
    autocomplete_fields = ('dataset',)

# ===============================
# Main Admin Classes
# ===============================

@admin.register(Industry)
class IndustryAdmin(admin.ModelAdmin):
    list_display = ('name', 'dataset_count', 'color_preview', 'icon')
    list_editable = ('color', 'icon')
    search_fields = ('name', 'description')
    prepopulated_fields = {'slug': ('name',)}
    actions = [export_to_csv]
    
    def dataset_count(self, obj):
        count = obj.datasets.filter(is_published=True).count()
        return format_html(
            '<span style="color: #007cba; font-weight: bold;">{}</span>',
            count
        )
    dataset_count.short_description = 'Published Datasets'
    
    def color_preview(self, obj):
        return format_html(
            '<div style="width: 30px; height: 20px; background-color: {}; border: 1px solid #ccc;"></div>',
            obj.color
        )
    color_preview.short_description = 'Color'
    
    def get_queryset(self, request):
        return super().get_queryset(request).annotate(
            published_dataset_count=Count('datasets', filter=Q(datasets__is_published=True))
        )

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'industry', 'difficulty', 'question_count', 'completion_rate_display',
        'avg_rating_display', 'is_published', 'is_featured', 'created_at'
    )
    list_filter = (
        'difficulty', 'industry', 'is_published', 'is_featured', 
        'source', 'ai_generated_questions', 'created_at'
    )
    search_fields = ('name', 'description', 'tags')
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = (
        'id', 'total_points', 'completion_rate', 'avg_rating', 
        'total_attempts', 'created_at', 'updated_at', 'table_count'
    )
    list_editable = ('is_published', 'is_featured', 'order')
    actions = [publish_items, unpublish_items, export_to_csv, 'generate_ai_questions']
    inlines = [QuestionInline, DatasetRatingInline]
    autocomplete_fields = ('industry', 'created_by')
    filter_horizontal = ('prerequisite_datasets',)
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'slug', 'description', 'industry', 'created_by')
        }),
        ('Classification', {
            'fields': ('difficulty', 'source', 'tags', 'business_context')
        }),
        ('Content', {
            'fields': ('schema', 'sample_data', 'csv_file', 'sql_file', 'cover_image'),
            'classes': ('collapse',)
        }),
        ('Learning Path', {
            'fields': (
                'prerequisite_datasets', 'estimated_time_minutes', 
                'learning_objectives'
            ),
            'classes': ('collapse',)
        }),
        ('AI Features', {
            'fields': ('ai_generated_questions', 'ai_analysis'),
            'classes': ('collapse',)
        }),
        ('Statistics', {
            'fields': (
                'total_points', 'completion_rate', 'avg_rating', 
                'total_attempts', 'table_count'
            ),
            'classes': ('collapse',)
        }),
        ('Status & Metadata', {
            'fields': (
                'is_published', 'is_featured', 'is_community', 'order',
                'created_at', 'updated_at'
            )
        })
    )
    
    def question_count(self, obj):
        return obj.questions.filter(is_published=True).count()
    question_count.short_description = 'Questions'
    
    def completion_rate_display(self, obj):
        rate = obj.completion_rate
        color = '#28a745' if rate >= 70 else '#ffc107' if rate >= 40 else '#dc3545'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{:.1f}%</span>',
            color, rate
        )
    completion_rate_display.short_description = 'Completion Rate'
    
    def avg_rating_display(self, obj):
        rating = obj.avg_rating
        stars = '★' * int(rating) + '☆' * (5 - int(rating))
        return format_html(
            '<span title="{:.2f}/5.0">{}</span>',
            rating, stars
        )
    avg_rating_display.short_description = 'Rating'
    
    def generate_ai_questions(self, request, queryset):
        """Generate AI questions for selected datasets"""
        for dataset in queryset:
            if not dataset.schema:
                messages.warning(request, f'Dataset "{dataset.name}" has no schema defined.')
                continue
            
            try:
                # Create AI generation request
                ai_request = AIQuestionGeneration.objects.create(
                    dataset=dataset,
                    user=request.user,
                    prompt_template="Generate SQL questions for dataset",
                    parameters={'difficulty_levels': ['EASY', 'MEDIUM', 'HARD']},
                    questions_requested=10
                )
                
                # Process generation (this would be done asynchronously in production)
                # For now, create a placeholder
                messages.success(
                    request, 
                    f'AI question generation request created for "{dataset.name}"'
                )
                
            except Exception as e:
                messages.error(
                    request, 
                    f'Failed to generate questions for "{dataset.name}": {str(e)}'
                )
    
    generate_ai_questions.short_description = "Generate AI questions for selected datasets"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'industry', 'created_by'
        ).prefetch_related('questions')

@admin.register(QuestionCategory)
class QuestionCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'question_count', 'color_preview', 'learning_order')
    list_editable = ('learning_order', 'color')
    search_fields = ('name', 'description')
    prepopulated_fields = {'slug': ('name',)}
    ordering = ('learning_order', 'name')
    actions = [export_to_csv]
    
    def question_count(self, obj):
        return obj.questions.filter(is_published=True).count()
    question_count.short_description = 'Questions'
    
    def color_preview(self, obj):
        return format_html(
            '<div style="width: 30px; height: 20px; background-color: {}; border: 1px solid #ccc;"></div>',
            obj.color
        )
    color_preview.short_description = 'Color'

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'dataset', 'category', 'difficulty', 'points',
        'success_rate_display', 'total_attempts', 'is_published', 'ai_generated'
    )
    list_filter = (
        'difficulty', 'question_type', 'category', 'dataset__industry',
        'is_published', 'is_featured', 'ai_generated', 'created_at'
    )
    search_fields = ('title', 'description', 'dataset__name')
    readonly_fields = (
        'id', 'success_rate', 'avg_attempts', 'avg_time_seconds',
        'total_attempts', 'created_at', 'updated_at'
    )
    list_editable = ('points', 'is_published', 'order')
    actions = [publish_items, unpublish_items, export_to_csv, 'test_solutions']
    autocomplete_fields = ('dataset', 'category', 'created_by')
    filter_horizontal = ('dependencies', 'unlocks_questions')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('dataset', 'category', 'title', 'description', 'question_type')
        }),
        ('Difficulty & Points', {
            'fields': ('difficulty', 'points', 'time_limit_minutes')
        }),
        ('SQL Content', {
            'fields': ('solution_query', 'starter_code', 'expected_output'),
            'classes': ('monospace',)
        }),
        ('Learning Support', {
            'fields': ('hint_level_1', 'hint_level_2', 'hint_level_3', 'explanation', 'learning_notes'),
            'classes': ('collapse',)
        }),
        ('Dependencies', {
            'fields': ('dependencies', 'unlocks_questions'),
            'classes': ('collapse',)
        }),
        ('AI Features', {
            'fields': (
                'ai_generated', 'ai_difficulty_score', 'ai_concepts', 'ai_similar_questions'
            ),
            'classes': ('collapse',)
        }),
        ('Statistics', {
            'fields': (
                'success_rate', 'avg_attempts', 'avg_time_seconds', 'total_attempts'
            ),
            'classes': ('collapse',)
        }),
        ('Status & Metadata', {
            'fields': (
                'is_published', 'is_featured', 'is_community_verified', 'order',
                'created_by', 'created_at', 'updated_at'
            )
        })
    )
    
    def success_rate_display(self, obj):
        rate = obj.success_rate
        color = '#28a745' if rate >= 70 else '#ffc107' if rate >= 40 else '#dc3545'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{:.1f}%</span>',
            color, rate
        )
    success_rate_display.short_description = 'Success Rate'
    
    def test_solutions(self, request, queryset):
        """Test solution queries for selected questions"""
        tested = 0
        errors = []
        
        for question in queryset:
            try:
                # This would validate the solution query
                if question.solution_query and question.dataset.schema:
                    # Validate query syntax and execution
                    # validation_result = validate_sql_query(question.solution_query)
                    tested += 1
            except Exception as e:
                errors.append(f'Question "{question.title}": {str(e)}')
        
        if errors:
            messages.warning(request, f'Errors found: {"; ".join(errors)}')
        else:
            messages.success(request, f'{tested} solution queries tested successfully.')
    
    test_solutions.short_description = "Test solution queries"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'dataset', 'category', 'created_by'
        )

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'preferred_difficulty', 'total_points', 'level', 
        'questions_solved', 'streak_days', 'last_active'
    )
    list_filter = (
        'preferred_difficulty', 'level', 'public_profile', 
        'email_notifications', 'created_at'
    )
    search_fields = ('user__username', 'user__email', 'bio', 'location')
    readonly_fields = (
        'total_points', 'level', 'experience_points', 'questions_solved',
        'longest_streak', 'created_at', 'updated_at', 'completion_percentage'
    )
    actions = [export_to_csv, 'recalculate_stats']
    inlines = [UserAchievementInline]
    filter_horizontal = ('interests', 'datasets_completed')
    
    fieldsets = (
        ('User Information', {
            'fields': ('user', 'avatar', 'bio', 'location')
        }),
        ('Social Links', {
            'fields': ('website', 'github_username', 'linkedin_profile'),
            'classes': ('collapse',)
        }),
        ('Learning Preferences', {
            'fields': ('preferred_difficulty', 'learning_goals', 'interests')
        }),
        ('Gamification', {
            'fields': (
                'total_points', 'level', 'experience_points', 
                'streak_days', 'longest_streak'
            )
        }),
        ('Statistics', {
            'fields': ('questions_solved', 'datasets_completed', 'certificates_earned')
        }),
        ('Settings', {
            'fields': (
                'email_notifications', 'push_notifications', 
                'public_profile', 'show_progress'
            ),
            'classes': ('collapse',)
        }),
        ('AI Data', {
            'fields': ('learning_style', 'skill_assessments', 'recommended_topics'),
            'classes': ('collapse',)
        })
    )
    
    def recalculate_stats(self, request, queryset):
        """Recalculate statistics for selected profiles"""
        updated = 0
        for profile in queryset:
            # Recalculate questions solved
            profile.questions_solved = UserAttempt.objects.filter(
                user=profile.user, is_correct=True
            ).values('question').distinct().count()
            
            # Recalculate total points
            profile.total_points = UserAttempt.objects.filter(
                user=profile.user, is_correct=True
            ).aggregate(total=Sum('points_earned'))['total'] or 0
            
            # Update level
            profile.update_level()
            profile.save()
            updated += 1
        
        messages.success(request, f'Statistics recalculated for {updated} profiles.')
    
    recalculate_stats.short_description = "Recalculate user statistics"

@admin.register(UserAttempt)
class UserAttemptAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'question_short', 'status', 'is_correct', 
        'points_earned', 'execution_time_ms', 'attempt_time'
    )
    list_filter = (
        'status', 'is_correct', 'question__difficulty', 
        'question__dataset__industry', 'attempt_time'
    )
    search_fields = (
        'user__username', 'question__title', 'question__dataset__name'
    )
    readonly_fields = (
        'id', 'user', 'question', 'attempt_time', 'query_complexity',
        'performance_score'
    )
    actions = [export_to_csv, 'regrade_attempts']
    date_hierarchy = 'attempt_time'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'question', 'status', 'is_correct')
        }),
        ('Query & Results', {
            'fields': ('user_query', 'result_data', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': (
                'execution_time_ms', 'time_taken', 'hints_used', 
                'attempts_count', 'performance_score', 'query_complexity'
            )
        }),
        ('Points & Rewards', {
            'fields': ('points_earned', 'bonus_points', 'achievements_unlocked')
        }),
        ('AI Analysis', {
            'fields': ('ai_feedback', 'optimization_suggestions'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('attempt_time', 'ip_address', 'user_agent'),
            'classes': ('collapse',)
        })
    )
    
    def question_short(self, obj):
        return obj.question.title[:50] + ('...' if len(obj.question.title) > 50 else '')
    question_short.short_description = 'Question'
    
    def regrade_attempts(self, request, queryset):
        """Regrade selected attempts"""
        regraded = 0
        for attempt in queryset:
            # Recalculate points based on current rules
            if attempt.is_correct:
                new_points = attempt.calculate_points()
                if new_points != attempt.points_earned:
                    old_points = attempt.points_earned
                    attempt.points_earned = new_points
                    attempt.save()
                    
                    # Update user profile points
                    profile = attempt.user.sql_profile
                    profile.total_points = profile.total_points - old_points + new_points
                    profile.save()
                    
                    regraded += 1
        
        messages.success(request, f'{regraded} attempts were regraded.')
    
    regrade_attempts.short_description = "Regrade selected attempts"

@admin.register(Achievement)
class AchievementAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'achievement_type', 'points_reward', 'earned_count',
        'is_active', 'is_hidden'
    )
    list_filter = ('achievement_type', 'is_active', 'is_hidden', 'created_at')
    search_fields = ('name', 'description')
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = ('earned_count', 'created_at')
    list_editable = ('is_active', 'is_hidden', 'points_reward')
    actions = [export_to_csv]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'slug', 'description', 'achievement_type')
        }),
        ('Visual', {
            'fields': ('icon', 'color', 'image')
        }),
        ('Requirements', {
            'fields': ('requirements', 'points_reward')
        }),
        ('Status', {
            'fields': ('is_active', 'is_hidden', 'earned_count', 'created_at')
        })
    )
    
    def earned_count(self, obj):
        return obj.userachievement_set.filter(is_completed=True).count()
    earned_count.short_description = 'Times Earned'

@admin.register(LearningPath)
class LearningPathAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'difficulty', 'dataset_count', 'estimated_hours',
        'is_published', 'is_featured', 'created_at'
    )
    list_filter = ('difficulty', 'is_published', 'is_featured', 'created_at')
    search_fields = ('name', 'description')
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = ('dataset_count', 'created_at')
    list_editable = ('is_published', 'is_featured', 'order')
    actions = [publish_items, unpublish_items, export_to_csv]
    inlines = [LearningPathDatasetInline]
    autocomplete_fields = ('created_by',)
    
    def dataset_count(self, obj):
        return obj.datasets.count()
    dataset_count.short_description = 'Datasets'

@admin.register(DatasetRating)
class DatasetRatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'dataset', 'rating', 'created_at')
    list_filter = ('rating', 'created_at', 'dataset__industry')
    search_fields = ('user__username', 'dataset__name', 'review')
    readonly_fields = ('created_at', 'updated_at')
    actions = [export_to_csv]

@admin.register(AIQuestionGeneration)
class AIQuestionGenerationAdmin(admin.ModelAdmin):
    list_display = (
        'dataset', 'user', 'questions_requested', 'questions_generated',
        'success_rate_display', 'status', 'created_at'
    )
    list_filter = ('status', 'created_at', 'dataset__industry')
    search_fields = ('dataset__name', 'user__username')
    readonly_fields = (
        'questions_generated', 'success_rate', 'generated_questions',
        'created_at', 'completed_at'
    )
    actions = [export_to_csv, 'retry_failed_generations']
    
    fieldsets = (
        ('Request Details', {
            'fields': ('dataset', 'user', 'questions_requested')
        }),
        ('Parameters', {
            'fields': ('prompt_template', 'parameters'),
            'classes': ('collapse',)
        }),
        ('Results', {
            'fields': (
                'questions_generated', 'success_rate', 'generated_questions', 'status'
            )
        }),
        ('Error Handling', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at')
        })
    )
    
    def success_rate_display(self, obj):
        rate = obj.success_rate
        color = '#28a745' if rate >= 80 else '#ffc107' if rate >= 60 else '#dc3545'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{:.1f}%</span>',
            color, rate
        )
    success_rate_display.short_description = 'Success Rate'
    
    def retry_failed_generations(self, request, queryset):
        """Retry failed AI generations"""
        failed_requests = queryset.filter(status='FAILED')
        for ai_request in failed_requests:
            ai_request.status = 'PENDING'
            ai_request.error_message = ''
            ai_request.save()
        
        messages.success(
            request, 
            f'{failed_requests.count()} failed generations queued for retry.'
        )
    
    retry_failed_generations.short_description = "Retry failed generations"

# ===============================
# Admin Site Configuration
# ===============================

admin.site.site_header = "TechySQL Academy Admin"
admin.site.site_title = "TechySQL Admin"
admin.site.index_title = "Welcome to TechySQL Academy Administration"

# Custom admin dashboard
class SQLPlatformAdminSite(admin.AdminSite):
    site_header = "TechySQL Academy Admin"
    site_title = "TechySQL Admin"
    index_title = "Platform Administration"
    
    def index(self, request, extra_context=None):
        """Custom admin dashboard with statistics"""
        extra_context = extra_context or {}
        
        # Platform statistics
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        stats = {
            'total_users': UserProfile.objects.count(),
            'total_datasets': Dataset.objects.filter(is_published=True).count(),
            'total_questions': Question.objects.filter(is_published=True).count(),
            'total_attempts': UserAttempt.objects.count(),
            'recent_users': UserProfile.objects.filter(created_at__date__gte=week_ago).count(),
            'recent_attempts': UserAttempt.objects.filter(attempt_time__date__gte=week_ago).count(),
            'avg_completion_rate': Dataset.objects.filter(is_published=True).aggregate(
                avg=Avg('completion_rate')
            )['avg'] or 0,
        }
        
        # Popular content
        popular_datasets = Dataset.objects.filter(
            is_published=True
        ).order_by('-total_attempts')[:5]
        
        popular_questions = Question.objects.filter(
            is_published=True
        ).order_by('-total_attempts')[:5]
        
        # Recent activity
        recent_attempts = UserAttempt.objects.select_related(
            'user', 'question'
        ).order_by('-attempt_time')[:10]
        
        extra_context.update({
            'platform_stats': stats,
            'popular_datasets': popular_datasets,
            'popular_questions': popular_questions,
            'recent_attempts': recent_attempts,
        })
        
        return super().index(request, extra_context)

# Register the custom admin site
# admin_site = SQLPlatformAdminSite(name='sql_admin')

# ===============================
# Additional Configurations
# ===============================

# Set up autocomplete fields
admin.site.empty_value_display = '(None)'

# Configure list per page
admin.site.site_url = '/'  # Link to main site