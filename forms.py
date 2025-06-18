# forms.py - Complete form definitions with validation
from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.utils.text import slugify
from django.conf import settings
from .models import (
    Dataset, Question, UserProfile, 
    LearningPath, DatasetRating, Industry
)
import os
import pandas as pd
import re

class DatasetUploadForm(forms.ModelForm):
    """
    Enhanced dataset upload form with CSV validation and automatic processing
    """
    csv_file = forms.FileField(
        label='Dataset CSV',
        help_text='Upload a CSV file (max 50MB). First row should contain column headers.',
        validators=[FileExtensionValidator(allowed_extensions=['csv'])],
        widget=forms.FileInput(attrs={
            'accept': '.csv',
            'class': 'file-upload'
        })
    )
    generate_sample_questions = forms.BooleanField(
        required=False,
        initial=True,
        help_text='Generate 3 sample questions automatically',
        widget=forms.CheckboxInput(attrs={
            'class': 'checkbox-input'
        })
    )
    industry = forms.ModelChoiceField(
        queryset=Industry.objects.all(),
        empty_label="Select Industry",
        help_text="Category for this dataset"
    )

    class Meta:
        model = Dataset
        fields = ['name', 'industry', 'difficulty', 'description', 'tags']
        widgets = {
            'name': forms.TextInput(attrs={
                'placeholder': 'Descriptive dataset name',
                'class': 'form-control'
            }),
            'description': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Brief description of the dataset contents',
                'class': 'form-control'
            }),
            'difficulty': forms.Select(attrs={
                'class': 'form-select'
            }),
            'tags': forms.TextInput(attrs={
                'placeholder': 'Comma-separated tags',
                'class': 'form-control'
            })
        }

    def clean_csv_file(self):
        csv_file = self.cleaned_data.get('csv_file')
        if not csv_file:
            raise ValidationError("CSV file is required")
        
        # Validate file size
        if csv_file.size > 50 * 1024 * 1024:  # 50MB
            raise ValidationError("File size exceeds 50MB limit")
        
        # Validate CSV structure
        try:
            # Read just the first few rows to validate
            df = pd.read_csv(csv_file.file, nrows=5)
            
            if df.empty:
                raise ValidationError("CSV file is empty")
                
            if len(df.columns) > 100:
                raise ValidationError("CSV has too many columns (max 100)")
                
            # Check for valid column names
            for col in df.columns:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                    raise ValidationError(
                        f"Invalid column name: '{col}'. Must start with letter/underscore "
                        "and contain only letters, numbers and underscores"
                    )
                    
        except pd.errors.EmptyDataError:
            raise ValidationError("CSV file appears to be empty")
        except pd.errors.ParserError:
            raise ValidationError("Could not parse CSV file - invalid format")
        except Exception as e:
            raise ValidationError(f"Error reading CSV: {str(e)}")
            
        return csv_file

    def clean_name(self):
        name = self.cleaned_data.get('name')
        if not name:
            raise ValidationError("Dataset name is required")
            
        # Check for duplicate slugs
        slug = slugify(name)
        if Dataset.objects.filter(slug=slug).exists():
            raise ValidationError("A dataset with a similar name already exists")
            
        return name

    def clean_tags(self):
        tags = self.cleaned_data.get('tags', [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        return tags[:10]  # Limit to 10 tags


class QuestionForm(forms.ModelForm):
    """
    Comprehensive question form with SQL validation and difficulty-based requirements
    """
    solution_query = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'sql-editor',
            'rows': 5,
            'placeholder': 'SELECT * FROM table...'
        }),
        help_text="The correct SQL solution for this question"
    )
    starter_code = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'sql-editor',
            'rows': 3,
            'placeholder': 'Optional starter code for the editor'
        })
    )

    class Meta:
        model = Question
        fields = [
            'title', 'description', 'dataset', 'category', 
            'difficulty', 'question_type', 'solution_query',
            'starter_code', 'hint_level_1', 'hint_level_2',
            'hint_level_3', 'explanation', 'points', 'time_limit_minutes'
        ]
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Clear question title'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Detailed question description'
            }),
            'dataset': forms.Select(attrs={
                'class': 'form-select'
            }),
            'category': forms.Select(attrs={
                'class': 'form-select'
            }),
            'difficulty': forms.Select(attrs={
                'class': 'form-select'
            }),
            'question_type': forms.Select(attrs={
                'class': 'form-select'
            }),
            'hint_level_1': forms.Textarea(attrs={
                'rows': 2,
                'class': 'form-control',
                'placeholder': 'Subtle hint (shown first)'
            }),
            'hint_level_2': forms.Textarea(attrs={
                'rows': 2,
                'class': 'form-control',
                'placeholder': 'More direct hint (shown second)'
            }),
            'hint_level_3': forms.Textarea(attrs={
                'rows': 2,
                'class': 'form-control',
                'placeholder': 'Detailed guidance (shown last)'
            }),
            'explanation': forms.Textarea(attrs={
                'rows': 4,
                'class': 'form-control',
                'placeholder': 'Detailed solution explanation'
            }),
            'points': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 1000
            }),
            'time_limit_minutes': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 120
            })
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Filter datasets based on user permissions
        if user and not user.is_superuser:
            self.fields['dataset'].queryset = Dataset.objects.filter(
                Q(created_by=user) | Q(is_published=True)
            
        # Set required hints based on difficulty
        if self.instance and self.instance.difficulty in ['HARD', 'EXPERT']:
            self.fields['hint_level_1'].required = True
            self.fields['hint_level_2'].required = True

    def clean_solution_query(self):
        query = self.cleaned_data.get('solution_query', '').strip()
        if not query:
            raise ValidationError("Solution query cannot be empty")
            
        # Basic SQL validation
        if ';' in query and query.count(';') > 1:
            raise ValidationError("Multiple statements not allowed")
            
        forbidden_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 
            'CREATE', 'INSERT', 'UPDATE', 'GRANT'
        ]
        for keyword in forbidden_keywords:
            if re.search(rf'\b{keyword}\b', query, re.IGNORECASE):
                raise ValidationError(f"Query contains forbidden operation: {keyword}")
                
        return query

    def clean(self):
        cleaned_data = super().clean()
        difficulty = cleaned_data.get('difficulty')
        points = cleaned_data.get('points', 10)
        
        # Validate points based on difficulty
        if difficulty == 'EASY' and points > 20:
            self.add_error('points', "Easy questions should have ≤ 20 points")
        elif difficulty == 'MEDIUM' and (points < 15 or points > 50):
            self.add_error('points', "Medium questions should have 15-50 points")
        elif difficulty == 'HARD' and points < 30:
            self.add_error('points', "Hard questions should have ≥ 30 points")
            
        # Validate time limits
        time_limit = cleaned_data.get('time_limit_minutes')
        if time_limit and difficulty == 'EASY' and time_limit > 15:
            self.add_error('time_limit_minutes', "Easy questions should have ≤ 15 minute limit")
            
        return cleaned_data


class UserProfileForm(forms.ModelForm):
    """
    Enhanced user profile form with learning preferences and validation
    """
    avatar = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control'
        }),
        help_text="Profile picture (square image recommended)"
    )
    learning_goals = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'rows': 3,
            'class': 'form-control',
            'placeholder': 'Your SQL learning objectives'
        }),
        help_text="Comma-separated list of goals"
    )

    class Meta:
        model = UserProfile
        fields = [
            'avatar', 'bio', 'location', 'website',
            'github_username', 'linkedin_profile',
            'preferred_difficulty', 'learning_goals',
            'interests', 'editor_theme', 'editor_font_size'
        ]
        widgets = {
            'bio': forms.Textarea(attrs={
                'rows': 3,
                'class': 'form-control',
                'placeholder': 'Tell us about yourself'
            }),
            'location': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'City, Country'
            }),
            'website': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://yourwebsite.com'
            }),
            'github_username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'GitHub username'
            }),
            'linkedin_profile': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'LinkedIn profile URL or ID'
            }),
            'preferred_difficulty': forms.Select(attrs={
                'class': 'form-select'
            }),
            'interests': forms.SelectMultiple(attrs={
                'class': 'form-select'
            }),
            'editor_theme': forms.Select(attrs={
                'class': 'form-select'
            }),
            'editor_font_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 10,
                'max': 24
            })
        }

    def clean_avatar(self):
        avatar = self.cleaned_data.get('avatar')
        if avatar:
            if avatar.size > 2 * 1024 * 1024:  # 2MB
                raise ValidationError("Avatar image too large (max 2MB)")
            ext = os.path.splitext(avatar.name)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                raise ValidationError("Unsupported image format")
        return avatar

    def clean_github_username(self):
        username = self.cleaned_data.get('github_username', '').strip()
        if username and not re.match(r'^[a-zA-Z\d](?:[a-zA-Z\d]|-(?=[a-zA-Z\d])){0,38}$', username):
            raise ValidationError("Invalid GitHub username format")
        return username

    def clean_learning_goals(self):
        goals = self.cleaned_data.get('learning_goals', '')
        if goals:
            goals = [goal.strip() for goal in goals.split(',') if goal.strip()]
            return goals[:5]  # Limit to 5 goals
        return []


class LearningPathForm(forms.ModelForm):
    """
    Form for creating and editing learning paths with dataset validation
    """
    cover_image = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control'
        }),
        help_text="Path cover image (1200x630px recommended)"
    )

    class Meta:
        model = LearningPath
        fields = [
            'name', 'description', 'difficulty', 
            'cover_image', 'estimated_hours'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Learning path title'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Detailed description of what this path covers'
            }),
            'difficulty': forms.Select(attrs={
                'class': 'form-select'
            }),
            'estimated_hours': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 100
            })
        }

    def clean_cover_image(self):
        image = self.cleaned_data.get('cover_image')
        if image:
            if image.size > 5 * 1024 * 1024:  # 5MB
                raise ValidationError("Cover image too large (max 5MB)")
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                raise ValidationError("Only JPG/PNG images allowed")
        return image


class DatasetRatingForm(forms.ModelForm):
    """
    Form for rating datasets with validation
    """
    class Meta:
        model = DatasetRating
        fields = ['rating', 'review']
        widgets = {
            'rating': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 5,
                'step': 1
            }),
            'review': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Share your experience with this dataset...'
            })
        }

    def clean_rating(self):
        rating = self.cleaned_data.get('rating')
        if rating not in [1, 2, 3, 4, 5]:
            raise ValidationError("Rating must be between 1 and 5")
        return rating


class IndustryForm(forms.ModelForm):
    """
    Form for managing industry categories
    """
    color = forms.CharField(
        widget=forms.TextInput(attrs={
            'type': 'color',
            'class': 'form-control form-control-color'
        }),
        help_text="Color for this industry category"
    )

    class Meta:
        model = Industry
        fields = ['name', 'description', 'icon', 'color']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Industry name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'Brief description'
            }),
            'icon': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'FontAwesome icon class (e.g. fa-database)'
            })
        }

    def clean_color(self):
        color = self.cleaned_data.get('color')
        if not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color):
            raise ValidationError("Invalid hex color code")
        return color

    def clean_icon(self):
        icon = self.cleaned_data.get('icon', '').strip()
        if icon and not icon.startswith('fa-'):
            raise ValidationError("Icon must be a FontAwesome class (e.g. 'fa-database')")
        return icon