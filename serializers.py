# serializers.py - Complete DRF serializers with validation and relationships
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import (
    Dataset, Question, UserProfile, UserAttempt,
    Achievement, UserAchievement, Industry,
    QuestionCategory, LearningPath, LearningPathDataset,
    DatasetRating, AIQuestionGeneration
)
from django.core.exceptions import ValidationError
from django.utils import timezone
import re

User = get_user_model()

class DynamicFieldsModelSerializer(serializers.ModelSerializer):
    """
    A ModelSerializer that takes an additional `fields` argument that
    controls which fields should be displayed.
    """
    def __init__(self, *args, **kwargs):
        fields = kwargs.pop('fields', None)
        super().__init__(*args, **kwargs)

        if fields is not None:
            allowed = set(fields)
            existing = set(self.fields)
            for field_name in existing - allowed:
                self.fields.pop(field_name)

class IndustrySerializer(DynamicFieldsModelSerializer):
    class Meta:
        model = Industry
        fields = ['id', 'name', 'slug', 'description', 'icon', 'color', 'dataset_count']
        read_only_fields = ['slug', 'dataset_count']

    def validate_color(self, value):
        if not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', value):
            raise serializers.ValidationError("Invalid hex color code")
        return value

    def validate_icon(self, value):
        if value and not value.startswith('fa-'):
            raise serializers.ValidationError("Icon must be a FontAwesome class (e.g. 'fa-database')")
        return value

class DatasetRatingSerializer(DynamicFieldsModelSerializer):
    user = serializers.StringRelatedField()
    dataset = serializers.StringRelatedField()

    class Meta:
        model = DatasetRating
        fields = ['id', 'user', 'dataset', 'rating', 'review', 'created_at']
        read_only_fields = ['user', 'dataset', 'created_at']

    def validate_rating(self, value):
        if value not in range(1, 6):
            raise serializers.ValidationError("Rating must be between 1 and 5")
        return value

class DatasetSerializer(DynamicFieldsModelSerializer):
    industry = IndustrySerializer(read_only=True)
    industry_id = serializers.PrimaryKeyRelatedField(
        queryset=Industry.objects.all(),
        source='industry',
        write_only=True
    )
    avg_rating = serializers.FloatField(read_only=True)
    question_count = serializers.IntegerField(read_only=True)
    is_completed = serializers.SerializerMethodField()
    ratings = DatasetRatingSerializer(many=True, read_only=True)

    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'slug', 'description', 'industry', 'industry_id',
            'difficulty', 'schema', 'sample_data', 'business_context',
            'cover_image', 'csv_file', 'avg_rating', 'question_count',
            'is_published', 'created_at', 'updated_at', 'is_completed',
            'tags', 'ratings', 'total_attempts'
        ]
        read_only_fields = [
            'slug', 'schema', 'sample_data', 'avg_rating',
            'question_count', 'created_at', 'updated_at'
        ]

    def get_is_completed(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.completed_by.filter(id=request.user.id).exists()
        return False

    def validate_tags(self, value):
        if len(value) > 10:
            raise serializers.ValidationError("Cannot add more than 10 tags")
        return value

    def validate_name(self, value):
        if not re.match(r'^[\w\s-]{3,100}$', value):
            raise serializers.ValidationError(
                "Name must be 3-100 characters long and can only contain letters, numbers, spaces and hyphens"
            )
        return value

class QuestionCategorySerializer(DynamicFieldsModelSerializer):
    class Meta:
        model = QuestionCategory
        fields = ['id', 'name', 'slug', 'description', 'icon', 'color']
        read_only_fields = ['slug']

class QuestionSerializer(DynamicFieldsModelSerializer):
    dataset = DatasetSerializer(read_only=True)
    dataset_id = serializers.PrimaryKeyRelatedField(
        queryset=Dataset.objects.filter(is_published=True),
        source='dataset',
        write_only=True
    )
    category = QuestionCategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=QuestionCategory.objects.all(),
        source='category',
        write_only=True
    )
    is_solved = serializers.SerializerMethodField()
    success_rate = serializers.FloatField(read_only=True)
    avg_time_seconds = serializers.FloatField(read_only=True)

    class Meta:
        model = Question
        fields = [
            'id', 'title', 'description', 'dataset', 'dataset_id',
            'category', 'category_id', 'difficulty', 'question_type',
            'solution_query', 'starter_code', 'hint_level_1', 'hint_level_2',
            'hint_level_3', 'explanation', 'points', 'time_limit_minutes',
            'is_published', 'is_solved', 'success_rate', 'avg_time_seconds',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'success_rate', 'avg_time_seconds', 'created_at', 'updated_at'
        ]
        extra_kwargs = {
            'solution_query': {'write_only': True}
        }

    def get_is_solved(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return UserAttempt.objects.filter(
                user=request.user,
                question=obj,
                is_correct=True
            ).exists()
        return False

    def validate(self, data):
        difficulty = data.get('difficulty', self.instance.difficulty if self.instance else None)
        points = data.get('points', self.instance.points if self.instance else None)
        
        # Validate points based on difficulty
        if difficulty and points:
            if difficulty == 'EASY' and points > 20:
                raise serializers.ValidationError(
                    {'points': "Easy questions should have ≤ 20 points"}
                )
            elif difficulty == 'MEDIUM' and (points < 15 or points > 50):
                raise serializers.ValidationError(
                    {'points': "Medium questions should have 15-50 points"}
                )
            elif difficulty == 'HARD' and points < 30:
                raise serializers.ValidationError(
                    {'points': "Hard questions should have ≥ 30 points"}
                )
        
        # Validate solution query
        solution_query = data.get('solution_query')
        if solution_query:
            forbidden = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
            if any(cmd in solution_query.upper() for cmd in forbidden):
                raise serializers.ValidationError(
                    {'solution_query': "Query contains forbidden operation"}
                )
        
        return data

class UserProfileSerializer(DynamicFieldsModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    full_name = serializers.SerializerMethodField()
    completion_percentage = serializers.FloatField(read_only=True)
    skill_mastery = serializers.SerializerMethodField()

    class Meta:
        model = UserProfile
        fields = [
            'id', 'username', 'email', 'full_name', 'avatar', 'bio',
            'location', 'website', 'github_username', 'linkedin_profile',
            'preferred_difficulty', 'learning_goals', 'interests',
            'total_points', 'level', 'experience_points', 'streak_days',
            'longest_streak', 'questions_solved', 'completion_percentage',
            'skill_mastery', 'editor_theme', 'editor_font_size',
            'public_profile', 'show_progress', 'last_active'
        ]
        read_only_fields = [
            'total_points', 'level', 'experience_points', 'streak_days',
            'longest_streak', 'questions_solved', 'completion_percentage',
            'last_active'
        ]

    def get_full_name(self, obj):
        return obj.user.get_full_name()

    def get_skill_mastery(self, obj):
        return obj.get_skill_mastery()

    def validate_github_username(self, value):
        if value and not re.match(r'^[a-zA-Z\d](?:[a-zA-Z\d]|-(?=[a-zA-Z\d])){0,38}$', value):
            raise serializers.ValidationError("Invalid GitHub username format")
        return value

    def validate_learning_goals(self, value):
        if len(value) > 5:
            raise serializers.ValidationError("Cannot have more than 5 learning goals")
        return value

class UserAttemptSerializer(DynamicFieldsModelSerializer):
    user = UserProfileSerializer(read_only=True)
    question = QuestionSerializer(read_only=True)
    question_id = serializers.PrimaryKeyRelatedField(
        queryset=Question.objects.filter(is_published=True),
        source='question',
        write_only=True
    )
    execution_time_seconds = serializers.SerializerMethodField()
    is_best_attempt = serializers.SerializerMethodField()

    class Meta:
        model = UserAttempt
        fields = [
            'id', 'user', 'question', 'question_id', 'user_query',
            'status', 'is_correct', 'execution_time_ms', 'execution_time_seconds',
            'time_taken', 'attempts_count', 'hints_used', 'result_data',
            'error_message', 'points_earned', 'performance_score',
            'query_complexity', 'ai_feedback', 'optimization_suggestions',
            'attempt_time', 'is_best_attempt'
        ]
        read_only_fields = [
            'user', 'status', 'execution_time_ms', 'time_taken',
            'attempts_count', 'points_earned', 'performance_score',
            'query_complexity', 'attempt_time'
        ]

    def get_execution_time_seconds(self, obj):
        return obj.execution_time_ms / 1000 if obj.execution_time_ms else None

    def get_is_best_attempt(self, obj):
        return UserAttempt.objects.filter(
            user=obj.user,
            question=obj.question,
            is_correct=True
        ).order_by('execution_time_ms').first() == obj

    def validate_user_query(self, value):
        if not value.strip():
            raise serializers.ValidationError("Query cannot be empty")
        
        forbidden = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        if any(cmd in value.upper() for cmd in forbidden):
            raise serializers.ValidationError("Query contains forbidden operation")
        
        if value.count(';') > 1:
            raise serializers.ValidationError("Multiple statements not allowed")
        
        return value

class AchievementSerializer(DynamicFieldsModelSerializer):
    is_unlocked = serializers.SerializerMethodField()
    progress = serializers.SerializerMethodField()

    class Meta:
        model = Achievement
        fields = [
            'id', 'name', 'slug', 'description', 'achievement_type',
            'icon', 'color', 'image', 'requirements', 'points_reward',
            'is_active', 'is_hidden', 'earned_count', 'is_unlocked', 'progress'
        ]
        read_only_fields = ['slug', 'earned_count']

    def get_is_unlocked(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return UserAchievement.objects.filter(
                user=request.user,
                achievement=obj,
                is_completed=True
            ).exists()
        return False

    def get_progress(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            try:
                user_achievement = UserAchievement.objects.get(
                    user=request.user,
                    achievement=obj
                )
                return user_achievement.progress
            except UserAchievement.DoesNotExist:
                return 0
        return 0

class UserAchievementSerializer(DynamicFieldsModelSerializer):
    achievement = AchievementSerializer(read_only=True)
    user = UserProfileSerializer(read_only=True)

    class Meta:
        model = UserAchievement
        fields = [
            'id', 'user', 'achievement', 'progress', 'is_completed',
            'unlocked_at', 'created_at', 'updated_at'
        ]
        read_only_fields = ['user', 'achievement', 'unlocked_at']

class LearningPathDatasetSerializer(DynamicFieldsModelSerializer):
    dataset = DatasetSerializer(read_only=True)
    dataset_id = serializers.PrimaryKeyRelatedField(
        queryset=Dataset.objects.filter(is_published=True),
        source='dataset',
        write_only=True
    )

    class Meta:
        model = LearningPathDataset
        fields = [
            'id', 'learning_path', 'dataset', 'dataset_id', 'order',
            'description', 'is_required', 'created_at'
        ]
        read_only_fields = ['learning_path', 'created_at']

class LearningPathSerializer(DynamicFieldsModelSerializer):
    created_by = UserProfileSerializer(read_only=True)
    datasets = LearningPathDatasetSerializer(
        source='learningpathdataset_set',
        many=True,
        read_only=True
    )
    progress = serializers.SerializerMethodField()
    dataset_ids = serializers.ListField(
        child=serializers.PrimaryKeyRelatedField(queryset=Dataset.objects.filter(is_published=True)),
        write_only=True,
        required=False
    )

    class Meta:
        model = LearningPath
        fields = [
            'id', 'name', 'slug', 'description', 'difficulty',
            'cover_image', 'estimated_hours', 'is_published',
            'is_featured', 'created_by', 'created_at', 'updated_at',
            'datasets', 'progress', 'dataset_ids'
        ]
        read_only_fields = ['slug', 'created_by', 'created_at', 'updated_at']

    def get_progress(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            from .utils import calculate_learning_path_progress
            return calculate_learning_path_progress(request.user, obj.id)
        return None

    def validate_dataset_ids(self, value):
        if len(value) > 20:
            raise serializers.ValidationError("Cannot add more than 20 datasets to a learning path")
        return value

    def create(self, validated_data):
        dataset_ids = validated_data.pop('dataset_ids', [])
        learning_path = LearningPath.objects.create(**validated_data)
        
        for order, dataset_id in enumerate(dataset_ids, start=1):
            LearningPathDataset.objects.create(
                learning_path=learning_path,
                dataset=dataset_id,
                order=order
            )
        
        return learning_path

    def update(self, instance, validated_data):
        dataset_ids = validated_data.pop('dataset_ids', None)
        instance = super().update(instance, validated_data)
        
        if dataset_ids is not None:
            # Clear existing and create new mappings
            instance.learningpathdataset_set.all().delete()
            for order, dataset_id in enumerate(dataset_ids, start=1):
                LearningPathDataset.objects.create(
                    learning_path=instance,
                    dataset=dataset_id,
                    order=order
                )
        
        return instance

class AIQuestionGenerationSerializer(DynamicFieldsModelSerializer):
    dataset = DatasetSerializer(read_only=True)
    user = UserProfileSerializer(read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = AIQuestionGeneration
        fields = [
            'id', 'dataset', 'user', 'prompt_template', 'parameters',
            'questions_requested', 'questions_generated', 'success_rate',
            'generated_questions', 'status', 'status_display', 'error_message',
            'created_at', 'completed_at'
        ]
        read_only_fields = [
            'questions_generated', 'success_rate', 'generated_questions',
            'status', 'error_message', 'created_at', 'completed_at'
        ]

    def validate_parameters(self, value):
        allowed_keys = ['difficulty_levels', 'categories', 'auto_create']
        if any(key not in allowed_keys for key in value.keys()):
            raise serializers.ValidationError("Invalid parameters provided")
        return value

# Specialized serializers for specific endpoints
class ExecuteQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True)
    question_id = serializers.UUIDField(required=True)

    def validate_query(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError("Query cannot be empty")
        
        forbidden = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        if any(cmd in value.upper() for cmd in forbidden):
            raise serializers.ValidationError("Query contains forbidden operation")
        
        if value.count(';') > 1:
            raise serializers.ValidationError("Multiple statements not allowed")
        
        return value

class GenerateQuestionsSerializer(serializers.Serializer):
    dataset_id = serializers.UUIDField(required=True)
    count = serializers.IntegerField(default=5, min_value=1, max_value=20)
    difficulty_levels = serializers.ListField(
        child=serializers.ChoiceField(choices=Question.DIFFICULTY_CHOICES),
        default=['EASY', 'MEDIUM', 'HARD']
    )
    categories = serializers.ListField(
        child=serializers.CharField(),
        required=False
    )

class ExplainQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True)
    dataset_id = serializers.UUIDField(required=False)

class DatasetUploadSerializer(serializers.ModelSerializer):
    csv_file = serializers.FileField(required=True)
    generate_sample_questions = serializers.BooleanField(default=True)

    class Meta:
        model = Dataset
        fields = [
            'name', 'industry', 'difficulty', 'description', 
            'csv_file', 'generate_sample_questions'
        ]

    def validate_csv_file(self, value):
        if value.size > 50 * 1024 * 1024:  # 50MB
            raise serializers.ValidationError("File size exceeds 50MB limit")
        return value