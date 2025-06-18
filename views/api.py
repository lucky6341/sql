# views/api.py - Complete API ViewSets with permissions and custom actions
from rest_framework import viewsets, status, mixins
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser, BasePermission
from rest_framework.pagination import PageNumberPagination
from django.db.models import Count, Q, F
from .models import (
    Dataset, Question, UserAttempt, 
    Achievement, UserAchievement, LearningPath,
    LearningPathDataset, DatasetRating, AIQuestionGeneration
)
from .serializers import (
    DatasetSerializer, QuestionSerializer, UserAttemptSerializer,
    AchievementSerializer, UserAchievementSerializer, LearningPathSerializer,
    LearningPathDatasetSerializer, DatasetRatingSerializer, AIQuestionGenerationSerializer,
    ExecuteQuerySerializer, GenerateQuestionsSerializer, ExplainQuerySerializer,
    DatasetUploadSerializer
)
from .utils import (
    execute_sql_query, check_query_correctness,
    calculate_query_performance, check_achievements
)
from .ai_utils import TechySQLAI
from .tasks import (
    analyze_user_attempt, generate_sample_questions,
    generate_ai_questions_task
)
import logging

logger = logging.getLogger(__name__)

class IsOwnerOrReadOnly(BasePermission):
    """
    Custom permission to only allow owners to edit their objects
    """
    def has_object_permission(self, request, view, obj):
        if request.method in ['GET', 'HEAD', 'OPTIONS']:
            return True
        return obj.created_by == request.user

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.annotate(
        question_count=Count('questions', filter=Q(questions__is_published=True))
    ).select_related('industry').prefetch_related('ratings')
    serializer_class = DatasetSerializer
    pagination_class = StandardResultsSetPagination
    lookup_field = 'id'

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [IsAuthenticated(), IsOwnerOrReadOnly()]
        return []

    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filtering
        difficulty = self.request.query_params.get('difficulty')
        industry = self.request.query_params.get('industry')
        search = self.request.query_params.get('search')
        
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
        if industry:
            queryset = queryset.filter(industry__slug=industry)
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(description__icontains=search) |
                Q(tags__icontains=search)
            )
            
        # Ordering
        order = self.request.query_params.get('order', '-total_attempts')
        if order in ['name', 'difficulty', 'created_at', 'total_attempts', 'avg_rating']:
            queryset = queryset.order_by(order)
            
        return queryset.filter(is_published=True)

    @action(detail=True, methods=['post'], serializer_class=DatasetRatingSerializer)
    def rate(self, request, id=None):
        dataset = self.get_object()
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        rating, created = DatasetRating.objects.update_or_create(
            user=request.user,
            dataset=dataset,
            defaults={
                'rating': serializer.validated_data['rating'],
                'review': serializer.validated_data.get('review', '')
            }
        )
        
        dataset.update_statistics()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'], serializer_class=DatasetUploadSerializer)
    def upload(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            dataset = Dataset.objects.create(
                name=serializer.validated_data['name'],
                industry=serializer.validated_data['industry'],
                difficulty=serializer.validated_data['difficulty'],
                description=serializer.validated_data['description'],
                created_by=request.user
            )
            
            # Process CSV and create database
            csv_file = request.FILES['csv_file']
            result = create_dataset_from_csv(
                csv_file=csv_file,
                dataset_name=dataset.name,
                industry_name=dataset.industry.name,
                difficulty=dataset.difficulty,
                created_by=request.user
            )
            
            if result['success']:
                dataset.schema = result['schema']
                dataset.sample_data = result['sample_data']
                dataset.save()
                
                if serializer.validated_data['generate_sample_questions']:
                    generate_sample_questions.delay(dataset.id)
                
                return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
            else:
                dataset.delete()
                return Response({'error': result['error']}, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            logger.error(f"Dataset upload failed: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class QuestionViewSet(viewsets.ModelViewSet):
    queryset = Question.objects.filter(is_published=True).select_related(
        'dataset', 'category', 'created_by'
    ).prefetch_related('dependencies')
    serializer_class = QuestionSerializer
    pagination_class = StandardResultsSetPagination
    lookup_field = 'id'

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [IsAuthenticated(), IsOwnerOrReadOnly()]
        return []

    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filtering
        dataset = self.request.query_params.get('dataset')
        difficulty = self.request.query_params.get('difficulty')
        category = self.request.query_params.get('category')
        unsolved = self.request.query_params.get('unsolved')
        
        if dataset:
            queryset = queryset.filter(dataset__id=dataset)
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
        if category:
            queryset = queryset.filter(category__slug=category)
        if unsolved and self.request.user.is_authenticated:
            solved_ids = UserAttempt.objects.filter(
                user=self.request.user,
                is_correct=True
            ).values_list('question_id', flat=True)
            queryset = queryset.exclude(id__in=solved_ids)
            
        return queryset

    @action(detail=True, methods=['post'], serializer_class=ExecuteQuerySerializer)
    def execute(self, request, id=None):
        question = self.get_object()
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            # Execute and validate query
            result = execute_sql_query(
                serializer.validated_data['query'],
                question.dataset.id,
                timeout=question.time_limit_minutes * 60 if question.time_limit_minutes else 30
            )
            
            if not result['success']:
                attempt = UserAttempt.objects.create(
                    user=request.user,
                    question=question,
                    user_query=serializer.validated_data['query'],
                    status='ERROR',
                    is_correct=False,
                    error_message=result['error'],
                    attempts_count=UserAttempt.objects.filter(
                        user=request.user,
                        question=question
                    ).count() + 1
                )
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            
            # Check correctness
            correctness = check_query_correctness(
                serializer.validated_data['query'],
                question.solution_query,
                question.dataset.id
            )
            
            # Create attempt record
            attempt = UserAttempt.objects.create(
                user=request.user,
                question=question,
                user_query=serializer.validated_data['query'],
                status='CORRECT' if correctness['correct'] else 'INCORRECT',
                is_correct=correctness['correct'],
                result_data=result['data'],
                performance_score=calculate_query_performance(
                    serializer.validated_data['query'],
                    result['data'],
                    result['execution_time']
                )['performance_score']
            )
            
            # Process results
            if correctness['correct']:
                # Trigger async analysis
                analyze_user_attempt.delay(attempt.id)
                
                # Prepare success response
                response_data = {
                    'correct': True,
                    'data': result['data'],
                    'columns': result['columns'],
                    'execution_time': result['execution_time'],
                    'points_earned': attempt.points_earned
                }
                return Response(response_data)
            else:
                # Prepare incorrect response with feedback
                response_data = {
                    'correct': False,
                    'data': result['data'],
                    'columns': result['columns'],
                    'differences': correctness.get('differences', []),
                    'ai_feedback': attempt.ai_feedback
                }
                return Response(response_data)
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def solution(self, request, id=None):
        question = self.get_object()
        if not request.user.has_perm('core.view_solution'):
            return Response(
                {'error': 'You do not have permission to view solutions'},
                status=status.HTTP_403_FORBIDDEN
            )
            
        return Response({
            'solution': question.solution_query,
            'explanation': question.explanation
        })

class UserAttemptViewSet(viewsets.ModelViewSet):
    serializer_class = UserAttemptSerializer
    pagination_class = StandardResultsSetPagination
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset = UserAttempt.objects.filter(user=self.request.user).select_related(
            'question', 'question__dataset', 'question__category'
        )
        
        # Filtering
        question = self.request.query_params.get('question')
        correct = self.request.query_params.get('correct')
        
        if question:
            queryset = queryset.filter(question__id=question)
        if correct:
            queryset = queryset.filter(is_correct=correct.lower() == 'true')
            
        return queryset.order_by('-attempt_time')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class AchievementViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Achievement.objects.filter(is_active=True)
    serializer_class = AchievementSerializer
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by type if specified
        achievement_type = self.request.query_params.get('type')
        if achievement_type:
            queryset = queryset.filter(achievement_type=achievement_type)
            
        return queryset

    @action(detail=False, methods=['get'])
    def mine(self, request):
        achievements = UserAchievement.objects.filter(
            user=request.user
        ).select_related('achievement').order_by('-unlocked_at')
        
        page = self.paginate_queryset(achievements)
        if page is not None:
            serializer = UserAchievementSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
            
        serializer = UserAchievementSerializer(achievements, many=True)
        return Response(serializer.data)

class LearningPathViewSet(viewsets.ModelViewSet):
    queryset = LearningPath.objects.filter(is_published=True).prefetch_related(
        'learningpathdataset_set__dataset'
    )
    serializer_class = LearningPathSerializer
    pagination_class = StandardResultsSetPagination
    lookup_field = 'id'

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [IsAuthenticated(), IsOwnerOrReadOnly()]
        return []

    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by difficulty
        difficulty = self.request.query_params.get('difficulty')
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
            
        return queryset

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    @action(detail=True, methods=['post'])
    def start(self, request, id=None):
        learning_path = self.get_object()
        profile = request.user.sql_profile
        
        # Mark as started (implementation depends on your model)
        profile.active_learning_path = learning_path
        profile.save()
        
        return Response({
            'status': 'started',
            'message': f'You have started the {learning_path.name} learning path'
        })

class AIQuestionGenerationViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet
):
    queryset = AIQuestionGeneration.objects.all()
    serializer_class = AIQuestionGenerationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return super().get_queryset().filter(user=self.request.user)

    def perform_create(self, serializer):
        generation = serializer.save(user=self.request.user)
        generate_ai_questions_task.delay(generation.id)