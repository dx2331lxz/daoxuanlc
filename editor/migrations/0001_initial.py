# Generated by Django 5.1.7 on 2025-03-29 13:25

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserPreference',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(max_length=255)),
                ('text_type', models.CharField(max_length=50)),
                ('preference_key', models.CharField(max_length=255)),
                ('preference_value', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'user_preferences',
                'indexes': [models.Index(fields=['user_id', 'text_type'], name='user_prefer_user_id_1e44e9_idx')],
            },
        ),
    ]
