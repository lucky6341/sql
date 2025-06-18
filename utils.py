# utils.py - Complete Implementation for TechySQL Academy
import sqlite3
import pandas as pd
import json
import re
import hashlib
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from django.conf import settings
from django.db import connections, transaction
from django.core.cache import cache
from django.utils import timezone
import pymysql
import logging
import requests
from contextlib import contextmanager
from .models import (
    Dataset, Question, UserProfile, UserAttempt, Achievement, 
    UserAchievement, QuestionCategory, Industry
)

logger = logging.getLogger(__name__)

# ===============================
# Database Connection Management
# ===============================

class DatabaseManager:
    """Manages multiple database connections for SQL execution"""
    
    def __init__(self):
        self.connections = {}
        self.lock = threading.Lock()
    
    def get_connection(self, dataset_id: str, db_type: str = 'sqlite') -> sqlite3.Connection:
        """Get or create a thread-safe database connection"""
        conn_key = f"{dataset_id}_{db_type}"
        
        with self.lock:
            if conn_key not in self.connections or not self._is_connection_alive(conn_key):
                if db_type == 'mysql':
                    self.connections[conn_key] = self._create_mysql_connection(dataset_id)
                else:
                    self.connections[conn_key] = self._create_sqlite_connection(dataset_id)
            
            return self.connections[conn_key]
    
    def _is_connection_alive(self, conn_key: str) -> bool:
        """Check if connection is still valid"""
        try:
            conn = self.connections[conn_key]
            if isinstance(conn, sqlite3.Connection):
                conn.execute("SELECT 1")
            else:  # MySQL connection
                conn.ping(reconnect=True)
            return True
        except:
            return False
    
    def _create_mysql_connection(self, dataset_id: str) -> pymysql.Connection:
        """Create MySQL connection for dataset"""
        try:
            return pymysql.connect(
                host=settings.DATASETS_DB_HOST,
                user=settings.DATASETS_DB_USER,
                password=settings.DATASETS_DB_PASSWORD,
                database=f"dataset_{dataset_id}",
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True
            )
        except Exception as e:
            logger.error(f"MySQL connection failed for dataset {dataset_id}: {e}")
            raise
    
    def _create_sqlite_connection(self, dataset_id: str) -> sqlite3.Connection:
        """Create SQLite connection for dataset (fallback)"""
        try:
            db_path = os.path.join(settings.MEDIA_ROOT, 'datasets', 'db', f'{dataset_id}.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            return conn
        except Exception as e:
            logger.error(f"SQLite connection failed for dataset {dataset_id}: {e}")
            raise
    
    def close_all(self):
        """Close all active connections"""
        with self.lock:
            for conn_key, conn in list(self.connections.items()):
                try:
                    if isinstance(conn, sqlite3.Connection):
                        conn.close()
                    else:
                        conn.close()
                except:
                    pass
                del self.connections[conn_key]

# Global database manager instance
db_manager = DatabaseManager()

# ===============================
# SQL Query Execution
# ===============================

@contextmanager
def dataset_connection(dataset_id: str):
    """Context manager for dataset database connections"""
    conn = None
    try:
        conn = db_manager.get_connection(dataset_id)
        yield conn
    finally:
        if conn:
            conn.rollback()  # Ensure no transactions are left open

def execute_sql_query(query: str, dataset_id: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute SQL query against dataset database with enhanced safety
    
    Args:
        query: SQL query to execute
        dataset_id: Dataset identifier
        timeout: Query timeout in seconds
    
    Returns:
        Dict with success status, data/error, and metadata
    """
    start_time = time.time()
    validation = validate_sql_query(query)
    
    if not validation['valid']:
        return {
            'success': False,
            'error': validation['error'],
            'execution_time': 0,
            'type': 'validation_error'
        }
    
    try:
        with dataset_connection(dataset_id) as conn:
            cursor = conn.cursor()
            
            # Set timeout and limits
            if isinstance(conn, sqlite3.Connection):
                cursor.execute(f"PRAGMA busy_timeout = {timeout * 1000};")
            else:  # MySQL
                cursor.execute(f"SET SESSION max_execution_time = {timeout * 1000};")
                cursor.execute("SET SESSION sql_select_limit = 1000;")
            
            # Execute query
            cursor.execute(query)
            
            # Fetch results with limits
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(1000)  # Limit to 1000 rows
                
                # Convert to list of dicts
                if isinstance(rows[0] if rows else None, dict):
                    data = list(rows)
                else:
                    data = [dict(zip(columns, row)) for row in rows]
            else:
                data = []
                columns = []
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return {
                'success': True,
                'data': data,
                'columns': columns,
                'row_count': len(data),
                'execution_time': execution_time,
                'type': 'success'
            }
    
    except pymysql.Error as e:
        return handle_database_error(e, start_time, 'mysql_error')
    except sqlite3.Error as e:
        return handle_database_error(e, start_time, 'sqlite_error')
    except Exception as e:
        return handle_database_error(e, start_time, 'execution_error')

def handle_database_error(error: Exception, start_time: float, error_type: str) -> Dict[str, Any]:
    """Standardize error response format"""
    execution_time = int((time.time() - start_time) * 1000)
    error_msg = str(error)
    
    # Classify common errors
    if "syntax error" in error_msg.lower():
        error_type = 'syntax_error'
    elif "no such table" in error_msg.lower():
        error_type = 'missing_table'
    elif "no such column" in error_msg.lower():
        error_type = 'missing_column'
    
    logger.error(f"Query execution failed: {error_msg}")
    
    return {
        'success': False,
        'error': error_msg,
        'execution_time': execution_time,
        'type': error_type
    }

def validate_sql_query(query: str) -> Dict[str, Any]:
    """
    Enhanced SQL query validation for security and syntax
    
    Args:
        query: SQL query to validate
    
    Returns:
        Dict with validation status and error message
    """
    if not query or not query.strip():
        return {'valid': False, 'error': 'Query cannot be empty'}
    
    query_upper = query.upper().strip()
    
    # Check for dangerous operations
    forbidden_keywords = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 
        'INSERT', 'UPDATE', 'GRANT', 'REVOKE', 'EXEC',
        'EXECUTE', 'XP_', 'SP_', 'SHUTDOWN', 'LOAD_FILE',
        'OUTFILE', 'DUMPFILE', 'INFORMATION_SCHEMA'
    ]
    
    for keyword in forbidden_keywords:
        if re.search(rf'\b{keyword}\b', query_upper):
            return {
                'valid': False, 
                'error': f'Operation "{keyword}" is not allowed'
            }
    
    # Check for multiple statements
    if query_upper.count(';') > 1:
        return {
            'valid': False,
            'error': 'Multiple statements are not allowed'
        }
    
    # Basic syntax validation
    if not query_upper.startswith('SELECT') and not query_upper.startswith('WITH'):
        return {
            'valid': False,
            'error': 'Only SELECT and WITH (CTE) queries are allowed'
        }
    
    # Check for balanced quotes and parentheses
    if (query.count('"') % 2 != 0 or 
        query.count("'") % 2 != 0 or 
        query.count("(") != query.count(")")):
        return {
            'valid': False,
            'error': 'Unbalanced quotes or parentheses'
        }
    
    return {'valid': True, 'error': None}

def check_query_correctness(user_query: str, solution_query: str, dataset_id: str) -> Dict[str, Any]:
    """
    Enhanced query correctness checking with result comparison
    
    Args:
        user_query: User's SQL query
        solution_query: Correct solution query
        dataset_id: Dataset identifier
    
    Returns:
        Dict with correctness status and detailed comparison
    """
    try:
        # Execute both queries
        user_result = execute_sql_query(user_query, dataset_id)
        solution_result = execute_sql_query(solution_query, dataset_id)
        
        if not user_result['success']:
            return {
                'correct': False,
                'error': user_result['error'],
                'type': 'execution_error',
                'user_result': None,
                'expected_result': None
            }
        
        if not solution_result['success']:
            return {
                'correct': False,
                'error': 'Solution query failed to execute',
                'type': 'solution_error',
                'user_result': None,
                'expected_result': None
            }
        
        # Compare results
        comparison = compare_query_results(
            user_result['data'], 
            solution_result['data'],
            user_result['columns'],
            solution_result['columns']
        )
        
        return {
            'correct': comparison['match'],
            'match_percentage': comparison['match_percentage'],
            'type': 'result_comparison',
            'user_result': user_result,
            'expected_result': solution_result,
            'differences': comparison.get('differences', []),
            'column_comparison': comparison.get('column_comparison', {})
        }
    
    except Exception as e:
        return {
            'correct': False,
            'error': f'Error comparing results: {str(e)}',
            'type': 'comparison_error',
            'user_result': None,
            'expected_result': None
        }

def compare_query_results(user_data: List[Dict], solution_data: List[Dict], 
                         user_columns: List[str], solution_columns: List[str]) -> Dict[str, Any]:
    """
    Compare query results with detailed analysis
    
    Args:
        user_data: User's query results
        solution_data: Expected results
        user_columns: Column names from user's query
        solution_columns: Column names from solution
    
    Returns:
        Dict with comparison results and analysis
    """
    # Initialize comparison result
    result = {
        'match': False,
        'match_percentage': 0,
        'row_count_match': len(user_data) == len(solution_data),
        'column_match': set(user_columns) == set(solution_columns),
        'differences': []
    }
    
    # Check column names
    if not result['column_match']:
        missing_columns = set(solution_columns) - set(user_columns)
        extra_columns = set(user_columns) - set(solution_columns)
        
        result['differences'].append({
            'type': 'column_mismatch',
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns)
        })
        return result
    
    # Check row count
    if not result['row_count_match']:
        result['differences'].append({
            'type': 'row_count_mismatch',
            'user_rows': len(user_data),
            'expected_rows': len(solution_data)
        })
        return result
    
    # Convert to comparable formats
    user_sorted = sorted([tuple(sorted(row.items())) for row in user_data])
    solution_sorted = sorted([tuple(sorted(row.items())) for row in solution_data])
    
    # Check exact match
    if user_sorted == solution_sorted:
        result['match'] = True
        result['match_percentage'] = 100
        return result
    
    # Calculate partial match percentage
    total_cells = len(solution_data) * len(solution_columns)
    matched_cells = 0
    
    # Find specific differences
    differences = []
    for i, (user_row, sol_row) in enumerate(zip(user_sorted, solution_sorted)):
        row_diff = {
            'row': i + 1,
            'column_differences': []
        }
        
        for (user_col, user_val), (sol_col, sol_val) in zip(user_row, sol_row):
            if user_col == sol_col and user_val == sol_val:
                matched_cells += 1
            else:
                row_diff['column_differences'].append({
                    'column': user_col,
                    'user_value': user_val,
                    'expected_value': sol_val
                })
        
        if row_diff['column_differences']:
            differences.append(row_diff)
    
    result['match_percentage'] = int((matched_cells / total_cells) * 100) if total_cells > 0 else 0
    result['differences'] = differences
    result['match'] = (result['match_percentage'] == 100)
    
    return result

# ===============================
# Dataset Management
# ===============================

def create_dataset_from_csv(csv_file_path: str, dataset_name: str, industry_name: str, 
                          difficulty: str = 'BEGINNER', created_by=None) -> Dict[str, Any]:
    """
    Create a new dataset from CSV file with enhanced validation
    
    Args:
        csv_file_path: Path to CSV file
        dataset_name: Name for the dataset
        industry_name: Industry category
        difficulty: Dataset difficulty level
        created_by: User creating the dataset
    
    Returns:
        Dict with creation status and dataset info
    """
    try:
        # Validate CSV file
        if not os.path.exists(csv_file_path):
            return {'success': False, 'error': 'CSV file not found'}
        
        if os.path.getsize(csv_file_path) > 50 * 1024 * 1024:  # 50MB limit
            return {'success': False, 'error': 'File size exceeds 50MB limit'}
        
        # Read CSV with chunking for large files
        chunks = pd.read_csv(csv_file_path, chunksize=10000)
        df = pd.concat(chunks)
        
        if df.empty:
            return {'success': False, 'error': 'CSV file is empty'}
        
        if len(df.columns) > 100:
            return {'success': False, 'error': 'CSV has too many columns (max 100)'}
        
        # Generate dataset ID
        dataset_id = hashlib.sha256(
            f"{dataset_name}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Analyze CSV structure
        schema_analysis = analyze_csv_structure(df)
        
        # Get or create industry
        industry, created = Industry.objects.get_or_create(
            name=industry_name,
            defaults={
                'description': f'Datasets related to {industry_name}',
                'icon': 'fa-industry',
                'color': '#4F46E5'
            }
        )
        
        # Create dataset record
        dataset = Dataset.objects.create(
            id=dataset_id,
            name=dataset_name,
            slug=slugify(dataset_name),
            description=f"Dataset containing {df.shape[0]} records with {df.shape[1]} attributes",
            industry=industry,
            difficulty=difficulty,
            schema=schema_analysis['schema'],
            sample_data=schema_analysis['sample_data'],
            business_context=generate_business_context(df, industry_name),
            source='UPLOAD',
            created_by=created_by,
            is_published=False
        )
        
        # Create database and load data
        success = create_dataset_database(dataset_id, df, schema_analysis['schema'])
        
        if success:
            return {
                'success': True,
                'dataset': dataset,
                'records_loaded': df.shape[0],
                'columns': df.shape[1]
            }
        else:
            dataset.delete()
            return {'success': False, 'error': 'Failed to create dataset database'}
    
    except Exception as e:
        logger.error(f"Error creating dataset from CSV: {e}")
        return {'success': False, 'error': str(e)}

def analyze_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Enhanced CSV structure analysis with data profiling
    
    Args:
        df: Pandas DataFrame
    
    Returns:
        Dict with schema and sample data
    """
    schema = {
        "tables": [{
            "name": "main_table",
            "columns": [],
            "primary_key": None,
            "row_count": len(df),
            "approx_size_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }]
    }
    
    # Analyze each column
    for column in df.columns:
        col_data = df[column].dropna()
        dtype = str(df[column].dtype)
        
        # Enhanced type detection
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                sql_type = 'INTEGER'
            else:
                sql_type = 'REAL'
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            sql_type = 'DATETIME'
            dtype = 'datetime64'
        else:
            max_len = col_data.astype(str).str.len().max()
            sql_type = f'VARCHAR({max(max_len, 1)})'
        
        # Data profiling
        stats = {
            "name": column,
            "type": sql_type,
            "pandas_dtype": dtype,
            "unique_values": col_data.nunique(),
            "null_count": df[column].isnull().sum(),
            "sample_values": col_data.head(3).tolist(),
            "stats": {}
        }
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(col_data):
            stats["stats"].update({
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std())
            })
        
        # Check for potential keys
        is_unique = col_data.nunique() == len(col_data)
        is_primary = (is_unique and 
                     not df[column].isnull().any() and
                     ('id' in column.lower() or column.lower().endswith('_id')))
        
        if is_primary and not schema["tables"][0]["primary_key"]:
            schema["tables"][0]["primary_key"] = column
            stats["is_primary_key"] = True
        
        schema["tables"][0]["columns"].append(stats)
    
    # Generate sample data (first 100 rows)
    sample_data = {
        "main_table": df.head(100).to_dict('records')
    }
    
    return {
        'schema': schema,
        'sample_data': sample_data,
        'profiling': {
            'column_count': len(df.columns),
            'row_count': len(df),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicate_rows': df.duplicated().sum()
        }
    }

def create_dataset_database(dataset_id: str, df: pd.DataFrame, schema: Dict[str, Any]) -> bool:
    """
    Create SQLite database for dataset and load data
    
    Args:
        dataset_id: Dataset identifier
        df: DataFrame with data
        schema: Database schema
    
    Returns:
        Success status
    """
    try:
        db_path = os.path.join(settings.MEDIA_ROOT, 'datasets', 'db', f'{dataset_id}.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create SQLite database
        conn = sqlite3.connect(db_path)
        df.to_sql('data', conn, if_exists='replace', index=False)
        
        # Add indexes based on schema analysis
        for col in schema['tables'][0]['columns']:
            if col.get('unique_values', 0) / schema['tables'][0]['row_count'] < 0.1:  # Low cardinality
                conn.execute(f"CREATE INDEX idx_{col['name']} ON data({col['name']})")
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Error creating dataset database: {e}")
        if conn:
            conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        return False

def generate_business_context(df: pd.DataFrame, industry: str) -> str:
    """
    Generate business context for dataset using AI/heuristics
    
    Args:
        df: DataFrame
        industry: Industry category
    
    Returns:
        Business context description
    """
    contexts = {
        'E-commerce': f"""
        This dataset contains {df.shape[0]} records from an e-commerce platform. 
        Columns include: {', '.join(df.columns[:5])}...
        Use this data to analyze purchasing patterns, customer behavior, and business metrics.
        """,
        'Healthcare': f"""
        This healthcare dataset contains {df.shape[0]} patient records with {df.shape[1]} attributes.
        Columns include: {', '.join(df.columns[:5])}...
        The data can be used to analyze treatment outcomes and patient demographics.
        """,
        'Finance': f"""
        This financial dataset contains {df.shape[0]} transaction records.
        Columns include: {', '.join(df.columns[:5])}...
        Analyze spending patterns, account performance, and financial trends.
        """
    }
    
    return contexts.get(industry, f"""
    This dataset contains {df.shape[0]} records with {df.shape[1]} different attributes.
    Columns include: {', '.join(df.columns[:5])}...
    Use this data to practice various SQL operations including filtering, joining, and aggregating.
    """).strip()

# ===============================
# AI Integration
# ===============================

def generate_ai_questions(dataset_id: str, count: int = 5, 
                         difficulty_levels: List[str] = None) -> Dict[str, Any]:
    """
    Generate AI-powered questions for a dataset
    
    Args:
        dataset_id: Dataset identifier
        count: Number of questions to generate
        difficulty_levels: List of difficulty levels
    
    Returns:
        Dict with generated questions
    """
    if difficulty_levels is None:
        difficulty_levels = ['EASY', 'MEDIUM', 'HARD']
    
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        schema = dataset.schema
        
        questions = []
        for difficulty in difficulty_levels:
            questions.extend(
                generate_questions_by_difficulty(schema, difficulty, count // len(difficulty_levels))
            )
        
        return {
            'success': True,
            'questions': questions[:count],
            'count': len(questions[:count])
        }
    
    except Exception as e:
        logger.error(f"AI question generation failed: {e}")
        return {'success': False, 'error': str(e)}

def generate_questions_by_difficulty(schema: Dict, difficulty: str, count: int) -> List[Dict]:
    """Generate questions based on difficulty level"""
    table = schema['tables'][0]
    columns = [col['name'] for col in table['columns']]
    numeric_cols = [col['name'] for col in table['columns'] 
                   if col['pandas_dtype'] in ['int64', 'float64']]
    
    questions = []
    
    if difficulty == 'EASY':
        # Basic SELECT queries
        questions.append({
            'text': f"Retrieve all columns from the {table['name']} table",
            'sql': f"SELECT * FROM {table['name']}",
            'difficulty': 'EASY',
            'category': 'SELECT'
        })
        
        # Column-specific queries
        for col in columns[:min(3, len(columns))]:
            questions.append({
                'text': f"Retrieve only the {col} column",
                'sql': f"SELECT {col} FROM {table['name']}",
                'difficulty': 'EASY',
                'category': 'SELECT'
            })
    
    elif difficulty == 'MEDIUM':
        # Filtering queries
        for col in columns[:min(3, len(columns))]:
            sample_val = table['columns'][columns.index(col)]['sample_values'][0]
            questions.append({
                'text': f"Find records where {col} = {sample_val}",
                'sql': f"SELECT * FROM {table['name']} WHERE {col} = {sample_val}",
                'difficulty': 'MEDIUM',
                'category': 'FILTER'
            })
        
        # Basic aggregations
        if numeric_cols:
            for col in numeric_cols[:min(2, len(numeric_cols))]:
                questions.append({
                    'text': f"Calculate the average of {col}",
                    'sql': f"SELECT AVG({col}) FROM {table['name']}",
                    'difficulty': 'MEDIUM',
                    'category': 'AGGREGATION'
                })
    
    elif difficulty == 'HARD':
        # Complex queries with GROUP BY
        if len(columns) >= 2 and numeric_cols:
            group_col = [c for c in columns if c not in numeric_cols][0]
            agg_col = numeric_cols[0]
            
            questions.append({
                'text': f"Group by {group_col} and calculate average {agg_col}",
                'sql': f"SELECT {group_col}, AVG({agg_col}) FROM {table['name']} GROUP BY {group_col}",
                'difficulty': 'HARD',
                'category': 'GROUPING'
            })
        
        # Subqueries
        if numeric_cols:
            questions.append({
                'text': f"Find records where {numeric_cols[0]} is above average",
                'sql': f"SELECT * FROM {table['name']} WHERE {numeric_cols[0]} > (SELECT AVG({numeric_cols[0]}) FROM {table['name']})",
                'difficulty': 'HARD',
                'category': 'SUBQUERY'
            })
    
    return questions[:count]

def generate_ai_feedback(user_query: str, solution_query: str, 
                        user_result: Dict, expected_result: Dict) -> Dict[str, Any]:
    """
    Generate AI-powered feedback for user queries
    
    Args:
        user_query: User's SQL query
        solution_query: Correct solution
        user_result: User's query results
        expected_result: Expected results
    
    Returns:
        Dict with feedback and suggestions
    """
    # First check for common errors
    common_errors = check_common_mistakes(user_query, solution_query)
    if common_errors:
        return common_errors
    
    # Generate detailed feedback
    feedback = {
        'summary': 'Your query produced results but needs improvement',
        'suggestions': [],
        'correctness': 0,
        'performance': analyze_query_performance(user_query, user_result['execution_time'])
    }
    
    # Compare column structure
    if set(user_result['columns']) != set(expected_result['columns']):
        feedback['suggestions'].append({
            'type': 'column_mismatch',
            'message': f"Expected columns: {', '.join(expected_result['columns'])}, Got: {', '.join(user_result['columns'])}"
        })
    
    # Compare row count
    if len(user_result['data']) != len(expected_result['data']):
        feedback['suggestions'].append({
            'type': 'row_count',
            'message': f"Expected {len(expected_result['data'])} rows, got {len(user_result['data'])}"
        })
    
    # Calculate correctness percentage
    comparison = compare_query_results(
        user_result['data'],
        expected_result['data'],
        user_result['columns'],
        expected_result['columns']
    )
    feedback['correctness'] = comparison['match_percentage']
    
    # Add specific differences
    if comparison['differences']:
        feedback['differences'] = comparison['differences'][:3]  # Show first 3 differences
    
    # Add optimization suggestions
    feedback['optimization'] = get_optimization_suggestions(user_query)
    
    return feedback

def check_common_mistakes(user_query: str, solution_query: str) -> Optional[Dict]:
    """Check for common SQL mistakes"""
    user_lower = user_query.lower()
    sol_lower = solution_query.lower()
    
    # Check for missing WHERE clause
    if 'where' in sol_lower and 'where' not in user_lower:
        return {
            'summary': 'Missing WHERE clause',
            'suggestion': 'Your query is missing a filtering condition. Try adding a WHERE clause.',
            'type': 'missing_where'
        }
    
    # Check for missing GROUP BY
    if 'group by' in sol_lower and 'group by' not in user_lower:
        return {
            'summary': 'Missing GROUP BY',
            'suggestion': 'Your query needs to group results. Try adding a GROUP BY clause.',
            'type': 'missing_group_by'
        }
    
    # Check for missing JOIN
    if 'join' in sol_lower and 'join' not in user_lower:
        return {
            'summary': 'Missing JOIN',
            'suggestion': 'Your query needs to join tables. Check if you need to add a JOIN clause.',
            'type': 'missing_join'
        }
    
    return None

# ===============================
# Performance Analysis
# ===============================

def calculate_query_performance(query: str, result_data: List[Dict], 
                              execution_time: int) -> Dict[str, Any]:
    """
    Enhanced query performance analysis
    
    Args:
        query: SQL query
        result_data: Query results
        execution_time: Execution time in milliseconds
    
    Returns:
        Performance metrics
    """
    metrics = {
        'execution_time': execution_time,
        'rows_returned': len(result_data),
        'query_complexity': analyze_query_complexity(query),
        'performance_score': 0,
        'optimization_suggestions': []
    }
    
    # Calculate performance score (0-100)
    base_score = 100
    
    # Time penalty
    if execution_time > 1000:  # Over 1 second
        time_penalty = min(50, (execution_time - 1000) // 100)
        base_score -= time_penalty
        metrics['optimization_suggestions'].append(
            f"Query took {execution_time}ms - consider optimizing"
        )
    
    # Complexity adjustment
    complexity = metrics['query_complexity']
    if complexity > 0.7:  # High complexity
        base_score += 5  # Bonus for complex queries
    elif complexity < 0.3:  # Low complexity
        base_score -= 5  # Penalty for simple queries that are slow
    
    # Row count adjustment
    if len(result_data) > 500:
        metrics['optimization_suggestions'].append(
            "Query returned many rows - consider adding LIMIT"
        )
    
    metrics['performance_score'] = max(0, min(100, base_score))
    metrics['optimization_suggestions'].extend(
        get_optimization_suggestions(query)
    )
    
    return metrics

def analyze_query_complexity(query: str) -> float:
    """
    Analyze SQL query complexity (0-1 scale)
    
    Args:
        query: SQL query
    
    Returns:
        Complexity score (0-1)
    """
    complexity = 0.0
    query_lower = query.lower()
    
    # Base complexity
    complexity += 0.1  # Minimum for any query
    
    # Joins
    join_types = ['join', 'inner join', 'left join', 'right join', 'full join']
    for join_type in join_types:
        complexity += query_lower.count(join_type) * 0.15
    
    # Subqueries
    complexity += query_lower.count('(select') * 0.2
    
    # Aggregations
    agg_functions = ['sum(', 'avg(', 'count(', 'max(', 'min(']
    complexity += sum(query_lower.count(func) * 0.1 for func in agg_functions)
    
    # GROUP BY
    if 'group by' in query_lower:
        complexity += 0.15
    
    # HAVING
    if 'having' in query_lower:
        complexity += 0.1
    
    # Window functions
    if 'over (' in query_lower:
        complexity += 0.25
    
    # CTEs
    if 'with ' in query_lower:
        complexity += 0.2
    
    # ORDER BY
    if 'order by' in query_lower:
        complexity += 0.05
    
    return min(1.0, complexity)

def get_optimization_suggestions(query: str) -> List[str]:
    """
    Get query optimization suggestions
    
    Args:
        query: SQL query
    
    Returns:
        List of optimization suggestions
    """
    suggestions = []
    query_lower = query.lower()
    
    # Check for SELECT *
    if 'select *' in query_lower:
        suggestions.append("Avoid SELECT * - specify columns needed")
    
    # Check for lack of WHERE on large tables
    if 'from' in query_lower and 'where' not in query_lower:
        suggestions.append("Add WHERE clause to filter rows early")
    
    # Check for complex ORDER BY without LIMIT
    if 'order by' in query_lower and 'limit' not in query_lower:
        suggestions.append("Add LIMIT with ORDER BY to reduce sorting cost")
    
    # Check for multiple similar subqueries
    if query_lower.count('(select') > 2:
        suggestions.append("Consider using CTEs (WITH clause) for repeated subqueries")
    
    return suggestions

# ===============================
# Achievement System
# ===============================

def check_achievements(user, attempt: UserAttempt) -> List[Dict[str, Any]]:
    """
    Check and award achievements for user attempt
    
    Args:
        user: User instance
        attempt: UserAttempt instance
    
    Returns:
        List of newly unlocked achievements
    """
    new_achievements = []
    profile = user.sql_profile
    
    # Get all active achievements user hasn't completed
    achievements = Achievement.objects.filter(
        is_active=True
    ).exclude(
        userachievement__user=user,
        userachievement__is_completed=True
    )
    
    for achievement in achievements:
        progress_update = calculate_achievement_progress(achievement, user, attempt)
        
        if progress_update['unlocked']:
            user_achievement, created = UserAchievement.objects.update_or_create(
                user=user,
                achievement=achievement,
                defaults={
                    'progress': 100.0,
                    'is_completed': True,
                    'unlocked_at': timezone.now()
                }
            )
            new_achievements.append({
                'id': achievement.id,
                'name': achievement.name,
                'description': achievement.description,
                'icon': achievement.icon,
                'points_reward': achievement.points_reward
            })
            
            # Award points
            profile.total_points += achievement.points_reward
            profile.experience_points += achievement.points_reward
            profile.save()
    
    return new_achievements

def calculate_achievement_progress(achievement: Achievement, user: User, 
                                 attempt: UserAttempt) -> Dict[str, Any]:
    """
    Calculate progress toward an achievement
    
    Args:
        achievement: Achievement to check
        user: User instance
        attempt: Current attempt
    
    Returns:
        Dict with progress info and unlock status
    """
    requirements = achievement.requirements
    achievement_type = achievement.achievement_type
    
    progress = {
        'unlocked': False,
        'new_progress': 0,
        'requirements_met': False
    }
    
    try:
        if achievement_type == 'SKILL':
            # Skill mastery achievements
            category = requirements.get('category')
            threshold = requirements.get('threshold', 80)
            
            mastery = user.sql_profile.get_skill_mastery().get(category, {})
            current = mastery.get('percentage', 0)
            
            progress['new_progress'] = min(100, int((current / threshold) * 100))
            progress['unlocked'] = current >= threshold
        
        elif achievement_type == 'STREAK':
            # Streak achievements
            days_required = requirements.get('days', 7)
            current_streak = user.sql_profile.streak_days
            
            progress['new_progress'] = min(100, int((current_streak / days_required) * 100))
            progress['unlocked'] = current_streak >= days_required
        
        elif achievement_type == 'COMPLETION':
            # Dataset/question completion
            if 'dataset_id' in requirements:
                # Specific dataset completion
                completed = UserAttempt.objects.filter(
                    user=user,
                    question__dataset_id=requirements['dataset_id'],
                    is_correct=True
                ).values('question').distinct().count()
                
                total = Question.objects.filter(
                    dataset_id=requirements['dataset_id'],
                    is_published=True
                ).count()
                
                progress['new_progress'] = min(100, int((completed / total) * 100)) if total > 0 else 0
                progress['unlocked'] = completed >= total
            
            else:
                # General completion
                completed = UserAttempt.objects.filter(
                    user=user,
                    is_correct=True
                ).values('question').distinct().count()
                
                threshold = requirements.get('threshold', 10)
                progress['new_progress'] = min(100, int((completed / threshold) * 100))
                progress['unlocked'] = completed >= threshold
        
        elif achievement_type == 'MILESTONE':
            # Points milestones
            threshold = requirements.get('points', 1000)
            current = user.sql_profile.total_points
            
            progress['new_progress'] = min(100, int((current / threshold) * 100))
            progress['unlocked'] = current >= threshold
        
        elif achievement_type == 'SPECIAL':
            # Special condition achievements
            if requirements.get('type') == 'perfect_no_hints':
                perfect_attempts = UserAttempt.objects.filter(
                    user=user,
                    is_correct=True,
                    hints_used=0
                ).count()
                
                threshold = requirements.get('count', 5)
                progress['new_progress'] = min(100, int((perfect_attempts / threshold) * 100))
                progress['unlocked'] = perfect_attempts >= threshold
        
        # Update progress if not already unlocked
        if not progress['unlocked']:
            user_achievement, created = UserAchievement.objects.get_or_create(
                user=user,
                achievement=achievement,
                defaults={'progress': progress['new_progress']}
            )
            
            if not created and user_achievement.progress < progress['new_progress']:
                user_achievement.progress = progress['new_progress']
                user_achievement.save()
    
    except Exception as e:
        logger.error(f"Error calculating achievement progress: {e}")
    
    return progress

# ===============================
# Utility Functions
# ===============================

def slugify(text: str) -> str:
    """Generate URL-safe slug from text"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text

def format_sql(query: str) -> str:
    """Format SQL query for display"""
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 
        'ORDER BY', 'LIMIT', 'JOIN', 'INNER', 'OUTER',
        'LEFT', 'RIGHT', 'ON', 'AS', 'AND', 'OR', 'NOT',
        'IN', 'BETWEEN', 'IS NULL', 'IS NOT NULL'
    ]
    
    for keyword in keywords:
        query = re.sub(rf'\b{keyword}\b', f'\n{keyword}', query, flags=re.IGNORECASE)
    
    return query.strip()

def get_query_columns(query: str) -> List[str]:
    """Extract column names from SQL query"""
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)
    if not select_match:
        return []
    
    select_part = select_match.group(1)
    columns = [col.strip().split()[-1] for col in select_part.split(',')]
    return [re.sub(r'^.*\.', '', col) for col in columns]  # Remove table prefixes

def calculate_learning_path_progress(user: User, learning_path_id: str) -> Dict[str, Any]:
    """
    Calculate user progress through a learning path
    
    Args:
        user: User instance
        learning_path_id: LearningPath ID
    
    Returns:
        Progress dictionary with completion stats
    """
    path = LearningPath.objects.get(id=learning_path_id)
    datasets = path.datasets.all()
    
    total_questions = 0
    completed_questions = 0
    completed_datasets = []
    
    for dataset in datasets:
        questions = dataset.questions.filter(is_published=True)
        total_questions += questions.count()
        
        solved = UserAttempt.objects.filter(
            user=user,
            question__in=questions,
            is_correct=True
        ).values('question').distinct().count()
        
        completed_questions += solved
        
        if solved == questions.count():
            completed_datasets.append(dataset.id)
    
    return {
        'total_datasets': datasets.count(),
        'completed_datasets': len(completed_datasets),
        'total_questions': total_questions,
        'completed_questions': completed_questions,
        'percentage': int((completed_questions / total_questions) * 100) if total_questions > 0 else 0
    }

def update_user_leaderboard():
    """Update cached leaderboard data"""
    leaderboard = UserProfile.objects.order_by('-total_points').select_related('user')[:100]
    cache.set('global_leaderboard', leaderboard, timeout=3600)  # Cache for 1 hour
    return leaderboard

def get_user_rank(user_id: int) -> Optional[int]:
    """Get user's global rank"""
    leaderboard = cache.get('global_leaderboard') or update_user_leaderboard()
    user_ids = [profile.user_id for profile in leaderboard]
    
    try:
        return user_ids.index(user_id) + 1
    except ValueError:
        return None