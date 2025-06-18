# tasks.py - Celery tasks for async processing
from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings
from django.core.mail import send_mail
from .models import (
    Dataset, Question, UserAttempt, 
    AIQuestionGeneration, UserProfile
)
from .utils import (
    create_dataset_from_csv, generate_ai_questions,
    check_query_correctness, update_user_leaderboard
)
from .ai_utils import TechySQLAI
import os
import pandas as pd
import time

logger = get_task_logger(__name__)

@shared_task(bind=True, max_retries=3)
def process_dataset_task(self, dataset_id, csv_path):
    """
    Async task to process dataset CSV and create database
    """
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        logger.info(f"Processing dataset: {dataset.name}")
        
        # Process CSV and create database
        result = create_dataset_from_csv(
            csv_path=csv_path,
            dataset_name=dataset.name,
            industry_name=dataset.industry.name,
            difficulty=dataset.difficulty,
            created_by=dataset.created_by
        )
        
        if result['success']:
            # Update dataset with generated schema
            dataset.schema = result['schema']
            dataset.sample_data = result['sample_data']
            dataset.save()
            
            # Generate sample questions if large dataset
            if result['records_loaded'] > 1000:
                generate_sample_questions.delay(dataset_id)
            
            logger.info(f"Successfully processed dataset: {dataset.name}")
            return {
                'status': 'SUCCESS',
                'dataset_id': str(dataset.id),
                'records_loaded': result['records_loaded']
            }
        else:
            logger.error(f"Failed to process dataset: {result['error']}")
            raise Exception(result['error'])
            
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise self.retry(exc=e, countdown=60 * 5)  # Retry after 5 minutes

@shared_task
def generate_sample_questions(dataset_id, count=3):
    """
    Generate sample questions for a dataset
    """
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        ai = TechySQLAI()
        questions = ai.generate_questions_from_schema(
            dataset.schema,
            count=count,
            difficulty_levels=['EASY']
        )
        
        created = []
        for q in questions:
            question = Question.objects.create(
                dataset=dataset,
                title=q['title'],
                description=q['description'],
                solution_query=q['solution'],
                difficulty=q['difficulty'],
                category=QuestionCategory.objects.filter(name__iexact=q['category']).first(),
                created_by=dataset.created_by,
                is_published=True
            )
            created.append(str(question.id))
        
        logger.info(f"Generated {len(created)} questions for dataset {dataset.name}")
        return {'status': 'SUCCESS', 'questions_created': created}
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return {'status': 'FAILED', 'error': str(e)}

@shared_task
def generate_ai_questions_task(generation_id):
    """
    Process AI question generation request
    """
    try:
        generation = AIQuestionGeneration.objects.get(id=generation_id)
        generation.status = 'PROCESSING'
        generation.save()
        
        ai = TechySQLAI()
        questions = ai.generate_questions_from_schema(
            generation.dataset.schema,
            count=generation.questions_requested,
            difficulty_levels=generation.parameters.get('difficulty_levels', ['EASY', 'MEDIUM', 'HARD']),
            categories=generation.parameters.get('categories')
        )
        
        # Store generated questions
        generation.generated_questions = questions
        generation.questions_generated = len(questions)
        generation.success_rate = (len(questions) / generation.questions_requested) * 100
        generation.status = 'COMPLETED'
        generation.completed_at = timezone.now()
        generation.save()
        
        # Optional: Create Question objects
        if generation.parameters.get('auto_create', False):
            create_questions_from_generation.delay(generation_id)
            
        return {'status': 'SUCCESS', 'questions_generated': len(questions)}
    except Exception as e:
        generation.status = 'FAILED'
        generation.error_message = str(e)
        generation.save()
        logger.error(f"AI generation failed: {str(e)}")
        return {'status': 'FAILED', 'error': str(e)}

@shared_task
def create_questions_from_generation(generation_id):
    """
    Create Question objects from AI generation results
    """
    generation = AIQuestionGeneration.objects.get(id=generation_id)
    created = []
    
    for q in generation.generated_questions:
        question = Question.objects.create(
            dataset=generation.dataset,
            title=q['title'],
            description=q['description'],
            solution_query=q['solution'],
            difficulty=q['difficulty'],
            category=QuestionCategory.objects.filter(name__iexact=q['category']).first(),
            created_by=generation.user,
            is_published=False,
            is_ai_generated=True
        )
        created.append(question.id)
    
    return {'status': 'SUCCESS', 'questions_created': created}

@shared_task
def analyze_user_attempt(attempt_id):
    """
    Perform deep analysis of user attempt (async)
    """
    attempt = UserAttempt.objects.get(id=attempt_id)
    
    # Generate detailed feedback
    feedback = generate_ai_feedback(
        attempt.user_query,
        attempt.question.solution_query,
        attempt.result_data,
        execute_sql_query(attempt.question.solution_query, attempt.question.dataset.id)
    )
    
    # Update attempt with analysis
    attempt.ai_feedback = feedback
    attempt.optimization_suggestions = feedback.get('optimization', [])
    attempt.save()
    
    # Update user skills profile
    update_user_skills.delay(attempt.user.id)
    
    return {'status': 'SUCCESS', 'attempt_id': str(attempt_id)}

@shared_task
def update_user_skills(user_id):
    """
    Update user's skill profile based on attempts
    """
    user = User.objects.get(id=user_id)
    profile = user.sql_profile
    
    # Recalculate skill mastery
    mastery = profile.get_skill_mastery(force_recalculate=True)
    profile.skill_assessments = mastery
    profile.save()
    
    # Check for achievements
    check_achievements_for_user.delay(user_id)
    
    return {'status': 'SUCCESS', 'user_id': str(user_id)}

@shared_task
def check_achievements_for_user(user_id):
    """
    Check and award achievements for a user
    """
    user = User.objects.get(id=user_id)
    profile = user.sql_profile
    new_achievements = []
    
    for achievement in Achievement.objects.filter(is_active=True):
        progress = calculate_achievement_progress(achievement, user)
        if progress['unlocked']:
            # Award achievement
            user_achievement, created = UserAchievement.objects.get_or_create(
                user=user,
                achievement=achievement,
                defaults={
                    'progress': 100,
                    'is_completed': True,
                    'unlocked_at': timezone.now()
                }
            )
            if created:
                new_achievements.append(achievement.name)
                profile.total_points += achievement.points_reward
                profile.save()
    
    if new_achievements:
        # Send notification
        send_achievement_notification.delay(
            user_id, 
            f"You unlocked {len(new_achievements)} new achievements!"
        )
    
    return {'status': 'SUCCESS', 'new_achievements': new_achievements}

@shared_task
def send_achievement_notification(user_id, message):
    """
    Send achievement notification to user
    """
    user = User.objects.get(id=user_id)
    if user.email and user.sql_profile.email_notifications:
        send_mail(
            'New Achievements Unlocked!',
            message,
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
            fail_silently=True
        )
    return {'status': 'SUCCESS', 'user_id': str(user_id)}

@shared_task
def update_leaderboard():
    """
    Periodic task to update leaderboard
    """
    leaderboard = update_user_leaderboard()
    return {'status': 'SUCCESS', 'count': len(leaderboard)}

@shared_task
def daily_maintenance():
    """
    Daily maintenance tasks
    """
    # Update streaks
    profiles = UserProfile.objects.filter(
        last_active__lt=timezone.now() - timedelta(days=1)
    )
    for profile in profiles:
        profile.update_streak()
    
    # Update leaderboard
    update_leaderboard.delay()
    
    # Clean up temp files
    clean_temp_files.delay()
    
    return {'status': 'SUCCESS', 'profiles_updated': profiles.count()}

@shared_task
def clean_temp_files():
    """
    Clean up temporary files
    """
    # Implementation would clean up old temporary files
    return {'status': 'SUCCESS'}