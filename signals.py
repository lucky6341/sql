# signals.py - Database signals and automatic processing
from django.db.models.signals import post_save, pre_save, post_delete
from django.dispatch import receiver
from django.db import transaction
from .models import (
    User, UserProfile, UserAttempt, Question,
    Dataset, Achievement, UserAchievement,
    LearningPathDataset
)
from .utils import check_achievements, update_user_leaderboard
from .tasks import (
    analyze_user_attempt, update_user_skills,
    check_achievements_for_user
)
import logging

logger = logging.getLogger(__name__)

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    Automatically create a profile when a new user is created
    """
    if created and not hasattr(instance, 'sql_profile'):
        UserProfile.objects.create(user=instance)
        logger.info(f"Created profile for new user: {instance.username}")

@receiver(post_save, sender=UserAttempt)
def handle_user_attempt(sender, instance, created, **kwargs):
    """
    Process user attempts and trigger related updates
    """
    if not created:
        return
        
    try:
        # Update question statistics
        instance.question.update_statistics()
        
        # Update dataset statistics
        instance.question.dataset.update_statistics()
        
        # Trigger async processing for correct attempts
        if instance.is_correct:
            transaction.on_commit(
                lambda: analyze_user_attempt.delay(instance.id)
            )
            transaction.on_commit(
                lambda: update_user_skills.delay(instance.user.id)
            )
            
    except Exception as e:
        logger.error(f"Error processing attempt {instance.id}: {str(e)}")

@receiver(post_save, sender=Question)
def update_question_dependencies(sender, instance, created, **kwargs):
    """
    Handle question dependency relationships
    """
    if not created and hasattr(instance, '_dependencies_updated'):
        return
        
    try:
        # Update unlocked questions when dependencies are met
        for question in instance.unlocks_questions.all():
            question.dependencies.remove(instance)
            
        instance._dependencies_updated = True
    except Exception as e:
        logger.error(f"Error updating question dependencies: {str(e)}")

@receiver(post_save, sender=Dataset)
def handle_dataset_update(sender, instance, created, **kwargs):
    """
    Process dataset updates and related objects
    """
    if not created:
        try:
            # Update all related questions when dataset is published
            if instance.is_published:
                instance.questions.filter(is_published=False).update(is_published=True)
        except Exception as e:
            logger.error(f"Error processing dataset update: {str(e)}")

@receiver(post_save, sender=UserAchievement)
def handle_achievement_unlock(sender, instance, created, **kwargs):
    """
    Process achievement unlocks and award points
    """
    if instance.is_completed and created:
        try:
            # Update user points
            profile = instance.user.sql_profile
            profile.total_points += instance.achievement.points_reward
            profile.save()
            
            # Update achievement stats
            instance.achievement.update_earned_count()
            
        except Exception as e:
            logger.error(f"Error processing achievement unlock: {str(e)}")

@receiver(post_save, sender=LearningPathDataset)
def update_learning_path_stats(sender, instance, created, **kwargs):
    """
    Update learning path statistics when datasets are added/removed
    """
    try:
        # Update estimated time based on datasets
        path = instance.learning_path
        total_time = sum(
            ds.dataset.estimated_time_minutes 
            for ds in path.learningpathdataset_set.all()
        ) / 60  # Convert to hours
        path.estimated_hours = max(1, round(total_time))
        path.save()
    except Exception as e:
        logger.error(f"Error updating learning path stats: {str(e)}")

@receiver(post_delete, sender=LearningPathDataset)
def update_learning_path_stats_on_delete(sender, instance, **kwargs):
    """
    Update learning path stats when datasets are removed
    """
    try:
        path = instance.learning_path
        total_time = sum(
            ds.dataset.estimated_time_minutes 
            for ds in path.learningpathdataset_set.all()
        ) / 60
        path.estimated_hours = max(1, round(total_time))
        path.save()
    except Exception as e:
        logger.error(f"Error updating learning path stats on delete: {str(e)}")

@receiver(pre_save, sender=UserProfile)
def update_user_level(sender, instance, **kwargs):
    """
    Automatically calculate user level based on experience points
    """
    if instance.experience_points >= (instance.level + 1) * 1000:
        instance.level = min(50, instance.experience_points // 1000)

@receiver(pre_save, sender=UserProfile)
def update_streak(sender, instance, **kwargs):
    """
    Update user login streak
    """
    today = timezone.now().date()
    if instance.last_active != today:
        if instance.last_active == today - timezone.timedelta(days=1):
            instance.streak_days += 1
            if instance.streak_days > instance.longest_streak:
                instance.longest_streak = instance.streak_days
        else:
            instance.streak_days = 1
        instance.last_active = today