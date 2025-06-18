# models.py - Complete Implementation for TechySQL Academy
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import (
    MinValueValidator, 
    MaxValueValidator, 
    FileExtensionValidator,
    validate_email
)
from django.core.exceptions import ValidationError
from django.db.models import Count, Sum, Avg, Q, F, Case, When, IntegerField
from django.utils import timezone
from django.urls import reverse
from django.core.files.storage import default_storage
import json
import uuid
import pandas as pd
from datetime import datetime, timedelta
import re
import os

# ----------------------
# Validators
# ----------------------

def validate_sql(value):
    """Validate SQL to prevent dangerous commands"""
    forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE']
    if any(cmd in value.upper() for cmd in forbidden):
        raise ValidationError("Potentially dangerous SQL command detected")
    
    if value.count('"') % 2 != 0 or value.count("'") % 2 != 0:
        raise ValidationError("Unbalanced quotes in SQL")
    if value.count(';') > 1:
        raise ValidationError("Multiple SQL statements not allowed")

def validate_hex_color(value):
    """Validate hex color codes"""
    if not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', value):
        raise ValidationError("Invalid hex color code")

def validate_json_array(value):
    """Validate that JSON field contains an array"""
    if not isinstance(value, list):
        raise ValidationError("Value must be a JSON array")

# ----------------------
# Abstract Base Models
# ----------------------

class TimestampedModel(models.Model):
    """Abstract model with created/updated timestamps"""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True

class OrderedModel(models.Model):
    """Abstract model for ordered items"""
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        abstract = True
        ordering = ['order']

# ----------------------
# Core Models
# ----------------------

class Industry(TimestampedModel):
    """Industry categories for datasets"""
    name = models.CharField(max_length=50, unique=True)
    slug = models.SlugField(max_length=60, unique=True)
    icon = models.CharField(max_length=50, help_text="FontAwesome icon class")
    description = models.TextField()
    color = models.CharField(
        max_length=7, 
        default="#4F46E5", 
        help_text="Hex color code",
        validators=[validate_hex_color]
    )
    
    class Meta:
        verbose_name_plural = "Industries"
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('industry_detail', args=[self.slug])

class Dataset(TimestampedModel, OrderedModel):
    """Enhanced dataset model with AI capabilities"""
    DIFFICULTY_CHOICES = [
        ('BEGINNER', 'Beginner'),
        ('INTERMEDIATE', 'Intermediate'), 
        ('ADVANCED', 'Advanced'),
        ('EXPERT', 'Expert')
    ]
    
    SOURCE_CHOICES = [
        ('UPLOAD', 'User Upload'),
        ('CURATED', 'Platform Curated'),
        ('COMMUNITY', 'Community Contributed'),
        ('AI_GENERATED', 'AI Generated')
    ]
    
    # Basic Information
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=250, unique=True)
    description = models.TextField()
    industry = models.ForeignKey(Industry, on_delete=models.PROTECT, related_name='datasets')
    
    # Metadata
    difficulty = models.CharField(max_length=15, choices=DIFFICULTY_CHOICES, default='BEGINNER')
    source = models.CharField(max_length=15, choices=SOURCE_CHOICES, default='CURATED')
    tags = models.JSONField(
        default=list, 
        blank=True, 
        help_text="Array of tags",
        validators=[validate_json_array]
    )
    
    # Content
    schema = models.JSONField(
        help_text="Database schema in JSON format",
        default=dict
    )
    sample_data = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Sample data preview"
    )
    business_context = models.TextField(
        blank=True, 
        help_text="Real-world business context"
    )
    
    # Files
    cover_image = models.ImageField(
        upload_to='dataset_covers/', 
        blank=True, 
        null=True,
        help_text="Cover image (1200x630px)"
    )
    csv_file = models.FileField(
        upload_to='datasets/csv/', 
        blank=True, 
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=['csv'])],
        help_text="Source CSV file"
    )
    sql_file = models.FileField(
        upload_to='datasets/sql/', 
        blank=True, 
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=['sql'])],
        help_text="Optional SQL file with schema"
    )
    
    # Learning Path
    prerequisite_datasets = models.ManyToManyField(
        'self', 
        symmetrical=False, 
        blank=True,
        help_text="Datasets that should be completed first"
    )
    estimated_time_minutes = models.PositiveIntegerField(
        default=60,
        validators=[MinValueValidator(5), MaxValueValidator(10080)],
        help_text="Estimated completion time in minutes"
    )
    learning_objectives = models.JSONField(
        default=list, 
        blank=True,
        help_text="List of learning objectives"
    )
    
    # Statistics (updated via signals)
    total_points = models.PositiveIntegerField(default=0)
    completion_rate = models.FloatField(default=0.0)
    avg_rating = models.FloatField(default=0.0)
    total_attempts = models.PositiveIntegerField(default=0)
    
    # Status
    is_published = models.BooleanField(default=False)
    is_featured = models.BooleanField(default=False)
    is_community = models.BooleanField(default=False)
    
    # AI Features
    ai_generated_questions = models.BooleanField(default=False)
    ai_analysis = models.JSONField(
        default=dict, 
        blank=True,
        help_text="AI-generated analysis of dataset"
    )
    
    # Metadata
    created_by = models.ForeignKey(
        User, 
        on_delete=models.PROTECT, 
        related_name='created_datasets'
    )
    
    class Meta:
        ordering = ['order', 'name']
        indexes = [
            models.Index(fields=['difficulty', 'industry']),
            models.Index(fields=['is_published', 'is_featured']),
            models.Index(fields=['created_at']),
        ]
        permissions = [
            ('can_curate_dataset', 'Can curate datasets'),
            ('can_publish_dataset', 'Can publish datasets'),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.get_difficulty_display()})"
    
    def clean(self):
        """Additional model validation"""
        if self.csv_file and not self.schema:
            try:
                self.schema = self.generate_schema()
            except Exception as e:
                raise ValidationError(f"Failed to generate schema: {str(e)}")
    
    def get_absolute_url(self):
        return reverse('dataset_detail', args=[str(self.id)])
    
    @property
    def table_count(self):
        """Get number of tables in schema"""
        try:
            return len(self.schema.get('tables', []))
        except (TypeError, AttributeError):
            return 0
    
    def generate_schema(self):
        """Generate schema from CSV file"""
        if not self.csv_file:
            return {}
            
        try:
            df = pd.read_csv(self.csv_file.path)
            schema = {
                "tables": [{
                    "name": "main_table",
                    "columns": [],
                    "row_count": len(df)
                }]
            }
            
            for column in df.columns:
                col_info = {
                    "name": column,
                    "type": str(df[column].dtype),
                    "unique_values": df[column].nunique(),
                    "null_count": df[column].isnull().sum(),
                    "sample_values": df[column].dropna().head(3).tolist()
                }
                schema["tables"][0]["columns"].append(col_info)
            
            return schema
        except Exception as e:
            raise ValidationError(f"CSV parsing error: {str(e)}")
    
    def update_statistics(self, save=True):
        """Update dataset statistics"""
        # Calculate total points
        self.total_points = self.questions.filter(
            is_published=True
        ).aggregate(
            total=Sum('points')
        )['total'] or 0
        
        # Calculate completion rate
        attempts = UserAttempt.objects.filter(question__dataset=self)
        if attempts.exists():
            correct_attempts = attempts.filter(is_correct=True).count()
            self.completion_rate = (correct_attempts / attempts.count()) * 100
        
        # Calculate average rating
        ratings = DatasetRating.objects.filter(dataset=self)
        if ratings.exists():
            self.avg_rating = ratings.aggregate(avg=Avg('rating'))['avg']
        
        if save:
            self.save(update_fields=[
                'total_points', 
                'completion_rate', 
                'avg_rating',
                'updated_at'
            ])
    
    def get_sample_questions(self, count=5):
        """Generate sample questions based on schema"""
        questions = []
        if not self.schema.get('tables'):
            return questions
            
        table = self.schema['tables'][0]
        
        # Basic SELECT questions
        questions.append({
            "text": f"Retrieve all records from the {table['name']} table",
            "sql": f"SELECT * FROM {table['name']}",
            "difficulty": "EASY",
            "category": "SELECT"
        })
        
        # Column-specific questions
        for col in table['columns'][:3]:
            questions.append({
                "text": f"Show all {col['name']} values from the dataset",
                "sql": f"SELECT {col['name']} FROM {table['name']}",
                "difficulty": "EASY",
                "category": "SELECT"
            })
        
        # Filtering questions
        if len(table['columns']) > 1:
            col = table['columns'][0]
            sample_value = col['sample_values'][0] if col['sample_values'] else 'value'
            questions.append({
                "text": f"Find records where {col['name']} is '{sample_value}'",
                "sql": f"SELECT * FROM {table['name']} WHERE {col['name']} = '{sample_value}'",
                "difficulty": "MEDIUM",
                "category": "FILTER"
            })
        
        return questions[:count]

class QuestionCategory(TimestampedModel, OrderedModel):
    """Categories for SQL questions"""
    name = models.CharField(max_length=50, unique=True)
    slug = models.SlugField(max_length=60, unique=True)
    description = models.TextField()
    icon = models.CharField(
        max_length=50, 
        help_text="FontAwesome icon class"
    )
    color = models.CharField(
        max_length=7, 
        default="#10B981",
        validators=[validate_hex_color]
    )
    
    class Meta:
        verbose_name_plural = "Question Categories"
        ordering = ['order', 'name']
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('category_detail', args=[self.slug])

class Question(TimestampedModel, OrderedModel):
    """Enhanced question model with AI capabilities"""
    DIFFICULTY_CHOICES = [
        ('EASY', 'Easy'),
        ('MEDIUM', 'Medium'),
        ('HARD', 'Hard'),
        ('EXPERT', 'Expert')
    ]
    
    QUESTION_TYPES = [
        ('PRACTICE', 'Practice'),
        ('CHALLENGE', 'Challenge'),
        ('ASSESSMENT', 'Assessment'),
        ('TUTORIAL', 'Tutorial')
    ]
    
    # Basic Information
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(
        Dataset, 
        on_delete=models.CASCADE, 
        related_name='questions'
    )
    category = models.ForeignKey(
        QuestionCategory, 
        on_delete=models.PROTECT, 
        related_name='questions'
    )
    
    title = models.CharField(max_length=300)
    description = models.TextField()
    question_type = models.CharField(
        max_length=15, 
        choices=QUESTION_TYPES, 
        default='PRACTICE'
    )
    difficulty = models.CharField(
        max_length=10, 
        choices=DIFFICULTY_CHOICES, 
        default='EASY'
    )
    
    # SQL Content
    solution_query = models.TextField(
        validators=[validate_sql],
        help_text="Correct SQL solution"
    )
    starter_code = models.TextField(
        blank=True,
        help_text="Initial code for editor"
    )
    expected_output = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Expected result in JSON format"
    )
    
    # Learning Support
    hint_level_1 = models.TextField(
        blank=True, 
        help_text="Basic hint"
    )
    hint_level_2 = models.TextField(
        blank=True, 
        help_text="Medium hint"
    )
    hint_level_3 = models.TextField(
        blank=True, 
        help_text="Detailed hint"
    )
    explanation = models.TextField(
        blank=True,
        help_text="Detailed solution explanation"
    )
    learning_notes = models.TextField(
        blank=True,
        help_text="Additional learning notes"
    )
    
    # Gamification
    points = models.PositiveIntegerField(
        default=10, 
        validators=[MinValueValidator(1), MaxValueValidator(1000)],
        help_text="Points awarded for correct answer"
    )
    time_limit_minutes = models.PositiveIntegerField(
        null=True, 
        blank=True,
        help_text="Optional time limit in minutes"
    )
    
    # Dependencies
    dependencies = models.ManyToManyField(
        'self', 
        symmetrical=False, 
        blank=True,
        help_text="Questions that should be completed first"
    )
    unlocks_questions = models.ManyToManyField(
        'self', 
        symmetrical=False, 
        blank=True, 
        related_name='unlocked_by',
        help_text="Questions unlocked by completing this one"
    )
    
    # AI Features
    ai_generated = models.BooleanField(
        default=False,
        help_text="Was this question AI-generated?"
    )
    ai_difficulty_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="AI-assessed difficulty score (0-1)"
    )
    ai_concepts = models.JSONField(
        default=list, 
        blank=True,
        help_text="List of SQL concepts tested"
    )
    ai_similar_questions = models.JSONField(
        default=list, 
        blank=True,
        help_text="Similar questions from other datasets"
    )
    
    # Statistics (updated via signals)
    success_rate = models.FloatField(default=0.0)
    avg_attempts = models.FloatField(default=0.0)
    avg_time_seconds = models.FloatField(default=0.0)
    total_attempts = models.PositiveIntegerField(default=0)
    
    # Status
    is_published = models.BooleanField(default=False)
    is_featured = models.BooleanField(default=False)
    is_community_verified = models.BooleanField(default=False)
    
    # Metadata
    created_by = models.ForeignKey(
        User, 
        on_delete=models.PROTECT, 
        related_name='created_questions'
    )
    
    class Meta:
        ordering = ['dataset', 'order', 'difficulty']
        unique_together = ['dataset', 'order']
        indexes = [
            models.Index(fields=['difficulty', 'category']),
            models.Index(fields=['is_published', 'is_featured']),
            models.Index(fields=['ai_generated']),
        ]
        permissions = [
            ('can_verify_question', 'Can verify community questions'),
        ]
    
    def __str__(self):
        return f"{self.dataset.name}: {self.title}"
    
    def clean(self):
        """Validate question data"""
        if not self.starter_code:
            self.starter_code = self.generate_starter_code()
        
        if not self.expected_output and self.solution_query and self.dataset.csv_file:
            self.expected_output = self.calculate_expected_output()
    
    def get_absolute_url(self):
        return reverse('question_detail', args=[str(self.id)])
    
    def generate_starter_code(self):
        """Generate default starter code based on dataset"""
        if not self.dataset.schema.get('tables'):
            return "SELECT * FROM table_name;"
            
        table = self.dataset.schema['tables'][0]
        return f"SELECT * FROM {table['name']} LIMIT 5;"
    
    def calculate_expected_output(self):
        """Execute solution query to get expected output"""
        from core.utils import execute_safe_query
        try:
            result = execute_safe_query(
                self.solution_query,
                self.dataset.csv_file.path
            )
            return {
                'columns': result['columns'],
                'rows': result['rows'][:100]  # Limit to 100 rows
            }
        except Exception as e:
            return {'error': str(e)}
    
    def update_statistics(self, save=True):
        """Update question statistics"""
        attempts = self.attempts.all()
        if attempts.exists():
            correct_attempts = attempts.filter(is_correct=True)
            self.success_rate = (correct_attempts.count() / attempts.count()) * 100
            self.total_attempts = attempts.count()
            
            # Calculate average attempts per user
            user_attempts = attempts.values('user').annotate(
                user_attempt_count=Count('id')
            ).aggregate(avg=Avg('user_attempt_count'))
            self.avg_attempts = user_attempts['avg'] or 0
            
            # Calculate average time
            timed_attempts = attempts.filter(time_taken__isnull=False)
            if timed_attempts.exists():
                self.avg_time_seconds = timed_attempts.aggregate(
                    avg=Avg('time_taken')
                )['avg']
        
        if save:
            self.save(update_fields=[
                'success_rate',
                'avg_attempts',
                'avg_time_seconds',
                'total_attempts',
                'updated_at'
            ])
    
    def generate_ai_feedback(self, user_query):
        """Generate AI feedback for user's query"""
        from core.ai_integration import get_sql_feedback
        return get_sql_feedback(
            question=self,
            user_query=user_query,
            correct_query=self.solution_query
        )

class UserProfile(TimestampedModel):
    """Extended user profile with learning analytics"""
    DIFFICULTY_PREFERENCE = [
        ('NOVICE', 'Novice'),
        ('BEGINNER', 'Beginner'),
        ('INTERMEDIATE', 'Intermediate'),
        ('ADVANCED', 'Advanced'),
        ('EXPERT', 'Expert')
    ]
    
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        related_name='sql_profile'
    )
    avatar = models.ImageField(
        upload_to='user_avatars/', 
        blank=True, 
        null=True
    )
    bio = models.TextField(blank=True)
    location = models.CharField(max_length=100, blank=True)
    website = models.URLField(blank=True)
    github_username = models.CharField(max_length=39, blank=True)
    linkedin_profile = models.CharField(max_length=100, blank=True)
    
    # Learning Preferences
    preferred_difficulty = models.CharField(
        max_length=15, 
        choices=DIFFICULTY_PREFERENCE, 
        default='INTERMEDIATE'
    )
    learning_goals = models.JSONField(
        default=list, 
        blank=True,
        help_text="User's learning objectives"
    )
    interests = models.ManyToManyField(
        Industry, 
        blank=True,
        help_text="Industries of interest"
    )
    
    # Gamification
    total_points = models.PositiveIntegerField(default=0)
    level = models.PositiveIntegerField(default=1)
    experience_points = models.PositiveIntegerField(default=0)
    streak_days = models.PositiveIntegerField(default=0)
    longest_streak = models.PositiveIntegerField(default=0)
    last_active = models.DateField(auto_now=True)
    
    # Statistics
    questions_solved = models.PositiveIntegerField(default=0)
    datasets_completed = models.ManyToManyField(
        Dataset, 
        blank=True,
        related_name='completed_by'
    )
    certificates_earned = models.PositiveIntegerField(default=0)
    
    # Settings
    email_notifications = models.BooleanField(default=True)
    push_notifications = models.BooleanField(default=False)
    public_profile = models.BooleanField(default=True)
    show_progress = models.BooleanField(default=True)
    
    # AI Data
    learning_style = models.JSONField(
        default=dict, 
        blank=True,
        help_text="AI-detected learning style"
    )
    skill_assessments = models.JSONField(
        default=dict, 
        blank=True,
        help_text="AI-generated skill assessments"
    )
    recommended_topics = models.JSONField(
        default=list, 
        blank=True,
        help_text="AI-recommended learning topics"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['total_points']),
            models.Index(fields=['level']),
            models.Index(fields=['last_active']),
        ]
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    def update_level(self):
        """Calculate user level based on experience points"""
        old_level = self.level
        self.level = min(50, max(1, self.experience_points // 1000))
        if self.level > old_level:
            return True
        return False
    
    def get_skill_mastery(self):
        """Calculate skill mastery percentages"""
        attempts = self.user.attempts.annotate(
            category_name=F('question__category__name')
        ).values('category_name').annotate(
            total=Count('id'),
            correct=Count(Case(
                When(is_correct=True, then=1),
                output_field=IntegerField()
            ))
        )
        
        mastery = {}
        for item in attempts:
            mastery[item['category_name']] = {
                'percentage': int((item['correct'] / item['total']) * 100) if item['total'] > 0 else 0,
                'total': item['total'],
                'correct': item['correct']
            }
        return mastery
    
    @property
    def completion_percentage(self):
        """Calculate overall completion percentage"""
        total_questions = Question.objects.filter(is_published=True).count()
        if total_questions == 0:
            return 0
        solved = UserAttempt.objects.filter(
            user=self.user,
            is_correct=True
        ).values('question').distinct().count()
        return int((solved / total_questions) * 100)
    
    def update_streak(self):
        """Update user's login streak"""
        today = timezone.now().date()
        if self.last_active == today - timedelta(days=1):
            self.streak_days += 1
            if self.streak_days > self.longest_streak:
                self.longest_streak = self.streak_days
        elif self.last_active != today:
            self.streak_days = 1
        self.last_active = today
        self.save()

class UserAttempt(TimestampedModel):
    """Track user attempts at solving questions"""
    STATUS_CHOICES = [
        ('CORRECT', 'Correct'),
        ('INCORRECT', 'Incorrect'),
        ('PARTIAL', 'Partially Correct'),
        ('TIMEOUT', 'Timed Out'),
        ('ERROR', 'Error')
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='attempts'
    )
    question = models.ForeignKey(
        Question, 
        on_delete=models.CASCADE, 
        related_name='attempts'
    )
    user_query = models.TextField()
    status = models.CharField(
        max_length=10, 
        choices=STATUS_CHOICES, 
        default='INCORRECT'
    )
    is_correct = models.BooleanField(default=False)
    execution_time_ms = models.PositiveIntegerField(
        help_text="Query execution time in milliseconds"
    )
    time_taken = models.PositiveIntegerField(
        help_text="Time taken to solve in seconds"
    )
    attempts_count = models.PositiveIntegerField(
        default=1,
        help_text="Attempt number for this question"
    )
    hints_used = models.PositiveIntegerField(default=0)
    result_data = models.JSONField(
        default=dict, 
        blank=True,
        help_text="Result data from query execution"
    )
    error_message = models.TextField(blank=True)
    points_earned = models.PositiveIntegerField(default=0)
    bonus_points = models.PositiveIntegerField(default=0)
    performance_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Query performance score (0-100)"
    )
    query_complexity = models.FloatField(
        null=True, 
        blank=True,
        help_text="Analyzed query complexity (0-1)"
    )
    ai_feedback = models.JSONField(
        default=dict, 
        blank=True,
        help_text="AI-generated feedback on the attempt"
    )
    optimization_suggestions = models.JSONField(
        default=list, 
        blank=True,
        help_text="AI-generated optimization suggestions"
    )
    achievements_unlocked = models.JSONField(
        default=list, 
        blank=True,
        help_text="Achievements unlocked with this attempt"
    )
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-attempt_time']
        indexes = [
            models.Index(fields=['user', 'question']),
            models.Index(fields=['is_correct']),
            models.Index(fields=['attempt_time']),
        ]
    
    def __str__(self):
        return f"{self.user.username}'s attempt on {self.question.title}"
    
    def calculate_points(self):
        """Calculate points earned for this attempt"""
        if not self.is_correct:
            return 0
        
        base_points = self.question.points
        
        # Apply time bonus/penalty
        time_factor = 1.0
        if self.question.time_limit_minutes:
            time_limit_seconds = self.question.time_limit_minutes * 60
            if self.time_taken > time_limit_seconds:
                time_factor = 0.5  # 50% penalty for overtime
            elif self.time_taken < time_limit_seconds / 2:
                time_factor = 1.2  # 20% bonus for fast completion
        
        # Apply hint penalty
        hint_penalty = 1.0 - (min(3, self.hints_used) * 0.1)  # 10% per hint
        
        # Apply performance bonus
        performance_bonus = 1.0
        if self.performance_score and self.performance_score > 80:
            performance_bonus = 1.1  # 10% bonus for high performance
        
        total_points = int(base_points * time_factor * hint_penalty * performance_bonus)
        
        # Ensure at least 1 point is awarded
        return max(1, total_points)

class Achievement(TimestampedModel):
    """Gamification achievements"""
    ACHIEVEMENT_TYPES = [
        ('SKILL', 'Skill Mastery'),
        ('STREAK', 'Learning Streak'),
        ('COMPLETION', 'Dataset Completion'),
        ('MILESTONE', 'Milestone'),
        ('SPECIAL', 'Special Achievement')
    ]
    
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=110, unique=True)
    description = models.TextField()
    achievement_type = models.CharField(
        max_length=15, 
        choices=ACHIEVEMENT_TYPES
    )
    icon = models.CharField(
        max_length=50, 
        help_text="FontAwesome icon class"
    )
    color = models.CharField(
        max_length=7, 
        default="#F59E0B",
        validators=[validate_hex_color]
    )
    image = models.ImageField(
        upload_to='achievement_images/', 
        blank=True, 
        null=True
    )
    requirements = models.JSONField(
        default=dict,
        help_text="Achievement requirements in JSON format"
    )
    points_reward = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    is_hidden = models.BooleanField(
        default=False,
        help_text="Hidden until unlocked"
    )
    earned_count = models.PositiveIntegerField(
        default=0,
        editable=False
    )
    
    class Meta:
        ordering = ['achievement_type', 'name']
    
    def __str__(self):
        return self.name
    
    def update_earned_count(self):
        """Update count of users who earned this achievement"""
        self.earned_count = self.userachievement_set.filter(
            is_completed=True
        ).count()
        self.save()

class UserAchievement(TimestampedModel):
    """Track user progress toward achievements"""
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='achievements'
    )
    achievement = models.ForeignKey(
        Achievement, 
        on_delete=models.CASCADE
    )
    progress = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    is_completed = models.BooleanField(default=False)
    unlocked_at = models.DateTimeField(
        null=True, 
        blank=True
    )
    
    class Meta:
        unique_together = ['user', 'achievement']
        ordering = ['-unlocked_at']
    
    def __str__(self):
        status = "Completed" if self.is_completed else "In Progress"
        return f"{self.user.username}'s {self.achievement.name} ({status})"
    
    def update_progress(self, new_progress):
        """Update achievement progress"""
        self.progress = min(100.0, max(0.0, new_progress))
        if self.progress >= 100.0 and not self.is_completed:
            self.is_completed = True
            self.unlocked_at = timezone.now()
            self.achievement.update_earned_count()
        self.save()

class LearningPath(TimestampedModel, OrderedModel):
    """Structured learning paths"""
    DIFFICULTY_CHOICES = [
        ('BEGINNER', 'Beginner'),
        ('INTERMEDIATE', 'Intermediate'),
        ('ADVANCED', 'Advanced'),
        ('EXPERT', 'Expert')
    ]
    
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=210, unique=True)
    description = models.TextField()
    difficulty = models.CharField(
        max_length=15, 
        choices=DIFFICULTY_CHOICES, 
        default='BEGINNER'
    )
    cover_image = models.ImageField(
        upload_to='learning_paths/', 
        blank=True, 
        null=True
    )
    estimated_hours = models.PositiveIntegerField(default=10)
    is_published = models.BooleanField(default=False)
    is_featured = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        User, 
        on_delete=models.PROTECT, 
        related_name='created_paths'
    )
    
    class Meta:
        ordering = ['order', 'name']
    
    def __str__(self):
        return self.name
    
    @property
    def dataset_count(self):
        return self.datasets.count()

class LearningPathDataset(TimestampedModel, OrderedModel):
    """Datasets in learning paths"""
    learning_path = models.ForeignKey(
        LearningPath, 
        on_delete=models.CASCADE, 
        related_name='datasets'
    )
    dataset = models.ForeignKey(
        Dataset, 
        on_delete=models.CASCADE
    )
    description = models.TextField(blank=True)
    is_required = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['order']
        unique_together = ['learning_path', 'dataset']
    
    def __str__(self):
        return f"{self.learning_path.name} - {self.dataset.name}"

class DatasetRating(TimestampedModel):
    """User ratings for datasets"""
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='dataset_ratings'
    )
    dataset = models.ForeignKey(
        Dataset, 
        on_delete=models.CASCADE, 
        related_name='ratings'
    )
    rating = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    review = models.TextField(blank=True)
    
    class Meta:
        unique_together = ['user', 'dataset']
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username}'s {self.rating}-star rating for {self.dataset.name}"

class AIQuestionGeneration(TimestampedModel):
    """Track AI-generated questions"""
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed')
    ]
    
    dataset = models.ForeignKey(
        Dataset, 
        on_delete=models.CASCADE, 
        related_name='ai_generations'
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='ai_question_requests'
    )
    prompt_template = models.TextField()
    parameters = models.JSONField(default=dict)
    questions_requested = models.PositiveIntegerField(default=5)
    questions_generated = models.PositiveIntegerField(default=0)
    generated_questions = models.JSONField(default=list)
    success_rate = models.FloatField(default=0.0)
    status = models.CharField(
        max_length=15, 
        choices=STATUS_CHOICES, 
        default='PENDING'
    )
    error_message = models.TextField(blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"AI Generation for {self.dataset.name} ({self.status})"

# ----------------------
# Signal Handlers
# ----------------------

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

@receiver(post_save, sender=UserAttempt)
def update_stats_on_attempt(sender, instance, **kwargs):
    """Update question and dataset stats when attempt is saved"""
    instance.question.update_statistics()
    instance.question.dataset.update_statistics()
    
    # Update user profile if correct attempt
    if instance.is_correct:
        profile = instance.user.sql_profile
        profile.total_points += instance.points_earned
        profile.experience_points += instance.points_earned
        profile.questions_solved = UserAttempt.objects.filter(
            user=instance.user,
            is_correct=True
        ).values('question').distinct().count()
        profile.update_level()
        profile.save()

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create profile when new user registers"""
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=DatasetRating)
def update_dataset_rating(sender, instance, **kwargs):
    """Update dataset rating stats when new rating added"""
    instance.dataset.update_statistics()