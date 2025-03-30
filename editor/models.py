from django.db import models

class UserPreference(models.Model):
    user_id = models.CharField(max_length=255)
    text_type = models.CharField(max_length=50)
    preference_key = models.CharField(max_length=255)
    preference_value = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'editor'
        db_table = 'user_preferences'
        indexes = [
            models.Index(fields=['user_id', 'text_type']),
        ]