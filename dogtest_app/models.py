from tkinter import CASCADE
from django.db import models
from django import forms

### 코드로 테스트하는 방법  유니테스트라던가..

# Create your models here.
class Serviceuser(models.Model) :
    # user_id = models.CharField(max_length=15, primary_key=True)
    # user_pw = models.CharField(max_length=225)
    userid = models.CharField(max_length=20, primary_key=True)
    password = models.CharField(max_length=225, null=False)
    created = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ['created']

# 사용자, 이미지 경로, 견종, 결과, 테스트시각
class Testresult(models.Model) :
    # userid =  models.ForeignKey(Serviceuser, on_delete=models.PROTECT)
    userid = models.CharField(max_length=20)
    image = models.CharField(max_length=100)
    dog_breed = models.CharField(max_length=100)
    testresult = models.CharField(max_length=10)
    like = models.IntegerField(blank=True,null=True)

## 프로젝트 폴더 > Settiongs.py에 
## MEDIA_URL = "/IMAGE/"  추가   (꼭 IMAGE로 할 필요 없음)

## pip install Pillow 해줘야 함
## 실제로 파일이 저장된느 경로를 설정하는 값
## - 기본적으로 BASE_DIR에는 manage.py가 위치한 경로가 지정되있는데 
## 그 하위경로에 'Image'라는 디렉토리 생성후 그 곳에 실제 image 파일을 저장하게 됨
## MEDIA_ROOT = os.path.join(BASE_DIR, "Image").replace('\\',"/")      