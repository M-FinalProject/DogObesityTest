from rest_framework import serializers
from .models import Serviceuser, Testresult

class ServiceuserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Serviceuser
        # fields = ['user_id','user_pw','created']
        fields = ['userid','password','created']

class TestresultSerializer(serializers.ModelSerializer):
    # userid = ServiceuserSerializer()
    class Meta:
        model = Testresult
        # fields = ['userid','image_path','dog_breed','test_result','created'],
        fields = ['userid','image','dog_breed','testresult','obesity_rate', 'like','created']


# ## 모델 예측 결과 반환 시
# class ResultSerializer(serializers.Serializer):
#     pre_rate = serializers.IntegerField()
#     cur_rate =  serializers.IntegerField()
#     cur_result = serializers.CharField(max_length = 20)