# Generated by Django 4.0.3 on 2022-04-06 12:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Serviceuser',
            fields=[
                ('userid', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('password', models.CharField(max_length=225)),
                ('created', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ['created'],
            },
        ),
        migrations.CreateModel(
            name='Testresult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('userid', models.CharField(max_length=20)),
                ('image', models.CharField(blank=True, max_length=100)),
                ('dog_breed', models.CharField(max_length=100)),
                ('testresult', models.CharField(max_length=10)),
                ('like', models.IntegerField(blank=True, null=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ['created'],
            },
        ),
    ]
