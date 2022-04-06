from collections import UserDict
from csv import unregister_dialect
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.core.files.storage import FileSystemStorage

import os, base64, io, cv2
import numpy as np
from PIL import Image
from uuid import uuid4
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from argon2 import PasswordHasher

from .models import Serviceuser, Testresult
from .serializers import ServiceuserSerializer, TestresultSerializer
from dog_models.predict import img_predict_keras, img_predict_torch
from dogtest.settings import MEDIA_URL


# Create your views here.
@csrf_exempt
def signup( request ):
    # GET요청이 들어오면 사용자가 입력한 ID 중복 여부 검사 - old한 방법
    # GET요청은 프론트엔드가 알려주는 스펙대로 다시 짜야해서 보류
    # if request.method == "GET" :   
    #     data = JSONParser().parse(request)
    #     desired_id = data['userid']
    #     # exist_id = Serviceuser.objects.get(user_id = desired_id)
    #     try : 
    #         if Serviceuser.objects.get(userid = desired_id) :   # 존재하면
    #             return HttpResponse(status=200)
    #     except:
    #         return HttpResponse(status=203)

    # POST요청이 들어오면 ID와 생성    
    if request.method == "POST" :
        # print(request)
        data = JSONParser().parse(request)
        data = data['user']
        data['password'] = PasswordHasher().hash(data['password'])
        serializer = ServiceuserSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            # return render( request, 'mainpage', status=201)  # 회원가입 성공 → mainpage로
            return JsonResponse(serializer.data, status=201)
        # return HttpResponseRedirect( reverse('signup'), status=400)   # 회원가입 실패 → signup page로
        return HttpResponse( status=400 )

@csrf_exempt
def login( request ):
    # ##### 그냥 임시로 구현 - 첫 페이지에 현재까지 눌러진 좋아요 수 표현
    # if request.method == 'GET' :
    #     total_count = Testresult.objects.all()
    #     like_count = Testresult.objects.filter(like=1)
    #     like_count = { 'total_count' : len(total_count) ,'like_count' : len(like_count) }
    #     return JsonResponse(like_count, status=200)
        
    if request.method == "POST":
        data = JSONParser().parse(request)
        data = data['user']
        input_id,  input_pw = data['userid'],  data['password']
        user_data = Serviceuser.objects.get(userid=input_id)
        try :  # DB에 저장된 id와 pw가 입력한 id와 pw가 일치한다면 status = 200
            if user_data.userid == input_id and PasswordHasher().verify(user_data.password, input_pw) :
                # return render( request, 'dogimage', status = 200)   # 로그인 성공 → dogimage page로
                return HttpResponse( status=200 )
        except : 
            # return HttpResponseRedirect( reverse('mainpage') ,status=400)   # 로그인 실패 → mainpage로
            return HttpResponse( status=400 )


## base64형식 decoding
def stringToRGB(base64_string):
    imgdata = base64.b64decode(base64_string)
    dataBytesIO = io.BytesIO(imgdata)
    image = Image.open(dataBytesIO)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


@csrf_exempt
def imageupload( request ):
    if request.method == "POST":
        request_data = JSONParser().parse(request)
        request_data = request_data['upload']

        userid = request_data['userid']
        img_base64 = request_data['image']
        dog_breed = request_data['dog_breed']

        decode_img = stringToRGB(img_base64)
        
        uuid_name = uuid4().hex
        img_name =f'{uuid_name}.jpg'

        dog_model = {
            'Chihuahua' : 'chihuahua_set1.h5',
            'Welsh Corgi' : 'cor_set4_B11_1.pth',
            'Beagle' : 'beagle_set5.h5',
            'Maltese' : '',
            'Retriever' : 'ret_set5_B_9_1.pth',
            'Dachshund' : 'dac_set9_b9_2.pth'
        }
        
        if dog_breed in ['Chihuahua', 'Beagle', 'Maltese'] :
            testresult = img_predict_keras(dog_breed, dog_model[dog_breed],decode_img, img_name)
            testresult = {'testresult' : testresult}
        elif dog_breed in ['Welsh Corgi','Retriever', 'Dachshund'] :
            testresult = img_predict_torch(dog_breed, dog_model[dog_breed],decode_img,img_name)

        testresult_data = {
            'userid' : userid,
            'image' : img_name,
            'dog_breed' : dog_breed, 
            'testresult' : testresult,
            'like' : 0 ,
        }

        serializer = TestresultSerializer(data=testresult_data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=200)
        else : 
            # print(serializer.errors)
            return HttpResponse( status=400 )

@csrf_exempt
def testresult(request):
    # POST방식으로 userid, dog_breed, image(이미지 이름), created(셍성날짜)
    if request.method == 'POST':
        data = JSONParser().parse(request)
        img_name = data['image']
        queryset = Testresult.objects.get(image=img_name)  
        queryset.like = 1
        queryset.save()
        return HttpResponse( status=200 )



# @csrf_exempt
# 사용자 id(userid), 견종(dog_breed), 업로드 된 이미지(image) 데이터를 multipart/form-data 형식으로 받아서 저리하고 json형식으로 주는 
# def imageupload( request ):
#     if request.method == "POST":
#         img_data = request.FILES['image']
#         dog_breed = request.POST['dog_breed']
#         # print(img_data, dog_breed)    

#         ext = img_data.name.split('.')[-1]
#         uuid_name = uuid4().hex
#         # print(f'{uuid_name}.{ext}')
#         img_name =f'{uuid_name}.{ext}'

#         # Image/ 경로에 저장
#         # fs = FileSystemStorage(location=f'Image/{dog_breed}/saveimg')
#         fs = FileSystemStorage(location=f'Image/{dog_breed}/saveimg', base_url='Image/{dog_breed}/')
#         fs.save(img_name, img_data)
        
#         dog_model = {
#             'Chihuahua' : '',
#             'Welsh Corgi' : 'corgi_model_4.h5',
#             'Beagle' : '',
#             'Maltese' : '',
#             'Retriever' : 'ret_set1_L6.pth',
#             'Dachshund' : ''
#         }
        
#         if dog_breed in ['Welsh Corgi'] :
#             testresult = img_predict_keras(dog_breed, dog_model[dog_breed],img_name)
#             testresult = {'testresult' : testresult}
#         elif dog_breed in ['Retriever'] :
#             testresult = img_predict_torch(dog_breed, dog_model[dog_breed],img_name)
#             testresult = {'testresult' : testresult}

#         new_data = {
#             'userid' : request.POST['userid'],
#             'image' : f'Image/{dog_breed}/saveimg/{img_name}',
#             'dog_breed' : request.POST['dog_breed'], 
#             'testresult' : testresult,
#         }

#         serializer = TestresultSerializer(data=new_data)
#         print(serializer.is_valid())
#         if serializer.is_valid():
#             serializer.save()
#             return JsonResponse(testresult, status=202)
#             # return render( request, 'testresult', status=201)
#         else : 
#             print(serializer.errors)
#             return HttpResponse(status=400)




################# 나중에 지워도 되는 코드 ###############
## 회원, 결과 값 생성이 잘 됐는지 살펴보기 위한 코드 (나중에 지울 코드)
# @csrf_exempt
# def user_list( request ) :
#     # GET요청이 들어오면 전체 address list를 내려주는  
#     if request.method == 'GET':
#         query_set = Serviceuser.objects.all()
#         serializer = ServiceuserSerializer(query_set, many=True)  # many옵션은 다수의 queryset형태를 serializer화 하고자 할 때 사용 
#         return JsonResponse(serializer.data, safe=False)
#     elif request.method == "DELETE":
#         data = JSONParser().parse(request)
#         data = data['user']
#         user_data = Serviceuser.objects.get(userid=data['userid'])
#         user_data.delete()
#         return HttpResponse(status=200)

# @csrf_exempt
# def testresult_list( request ) :
#     # GET요청이 들어오면 전체 address list를 내려주는  
#     if request.method == 'GET':
#         query_set = Testresult.objects.all()
#         serializer = TestresultSerializer(query_set, many=True)  # many옵션은 다수의 queryset형태를 serializer화 하고자 할 때 사용 
#         return JsonResponse(serializer.data, safe=False)
#     elif request.method == "DELETE":
#         data = JSONParser().parse(request)
#         user_data = Testresult.objects.filter(userid=data['userid'])
#         user_data.delete()
#         return HttpResponse(status=200)

# @cstf_exempt
