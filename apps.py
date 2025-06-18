from django.apps import AppConfig

class SqlCodeEditorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sql_code_editor'
    
    def ready(self):
        import sql_code_editor.signals  # noqa