# ai_utils.py - New file

import requests
import os
from django.conf import settings

class AIQuestionGenerator:
    """Handles AI question generation using free APIs"""
    
    def __init__(self):
        self.huggingface_url = "https://api-inference.huggingface.co/models/"
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        
    def generate_from_schema(self, schema, count=5):
        """Generate questions from dataset schema"""
        prompt = f"""
        Generate {count} SQL practice questions based on this database schema:
        {json.dumps(schema, indent=2)}
        
        For each question provide:
        - question_text: The question prompt
        - difficulty: BEGINNER, INTERMEDIATE or ADVANCED
        - category: SELECT, FILTER, JOIN, GROUP BY, etc.
        - solution: The correct SQL query
        """
        
        try:
            if settings.USE_GEMINI:
                return self._generate_with_gemini(prompt)
            else:
                return self._generate_with_huggingface(prompt)
        except Exception as e:
            return self._generate_fallback_questions(schema, count)
    
    def _generate_with_gemini(self, prompt):
        """Use Gemini API for question generation"""
        url = f"{self.gemini_url}gemini-pro:generateContent?key={settings.GEMINI_API_KEY}"
        response = requests.post(url, json={
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        })
        response.raise_for_status()
        return self._parse_ai_response(response.json())
    
    def _generate_with_huggingface(self, prompt):
        """Use Hugging Face API for question generation"""
        headers = {"Authorization": f"Bearer {settings.HF_API_KEY}"}
        response = requests.post(
            f"{self.huggingface_url}gpt2",
            headers=headers,
            json={"inputs": prompt}
        )
        response.raise_for_status()
        return self._parse_ai_response(response.json())
    
    def _generate_fallback_questions(self, schema, count):
        """Fallback question generation when AI fails"""
        questions = []
        table = schema['tables'][0]
        
        # Basic select questions
        questions.append({
            'question_text': f"Select all columns from {table['name']}",
            'difficulty': 'BEGINNER',
            'category': 'SELECT',
            'solution': f"SELECT * FROM {table['name']}"
        })
        
        # Add more fallback questions...
        return questions[:count]
    
    def _parse_ai_response(self, response):
        """Parse AI response into structured questions"""
        # Implementation depends on API response format
        pass