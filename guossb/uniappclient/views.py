from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from shibie import detect_img
import base64
import numpy as np
import cv2
import torch
import cv2
import json
#测试函数

def index(request):
    return HttpResponse("hello world")

#GET测试函数
def getces(request):
    title = request.GET.get('title')
    print(title)
    data = {
        "code":200,
        "title":"加几个字"+title
    }
    return JsonResponse(data)


#多图片识别api
def duoiphone(request):
    data = {}
    imgdata = request.POST.get('image')
    # print(imgdata)
    files = json.loads(imgdata)
    imgs = []
    counts = []
    for i in files:
        img0 = base64.b64decode(i)
        img1 = np.frombuffer(img0,np.uint8)
        img2 = cv2.imdecode(img1,cv2.COLOR_RGB2BGR)
        count,im6 = detect_img(img2)
        image = cv2.imencode('.jpg',im6)[1]
        base64img = str(base64.b64encode(image))[2:-1]
        imgs.append(base64img)
        counts.append(count)
    print('--------------------')
    # print(base64img)
    # txtwenj = open('/guoshusibieXM/guossb/uniappclient/aaa.txt')
    # txtwenj.write(base64img)
    # with open('uniappclient/base.txt','wb') as f:
    #     f.write(base64img.encode())
    data = {
        "img":imgs,
        "count":counts
    }
    return JsonResponse(data)

def oneiphone(request):
    data = {}
    imgdata = request.POST.get('image')
    img0 = base64.b64decode(imgdata)
    img1 = np.frombuffer(img0,np.uint8)
    img2 = cv2.imdecode(img1,cv2.COLOR_RGB2BGR)
    count,im6 = detect_img(img2)
    image = cv2.imencode('.jpg',im6)[1]
    base64img = str(base64.b64encode(image))[2:-1]
    print('--------------------')
    # print(base64img)
    # txtwenj = open('/guoshusibieXM/guossb/uniappclient/aaa.txt')
    # txtwenj.write(base64img)
    # with open('uniappclient/base.txt','wb') as f:
    #     f.write(base64img.encode())
    data = {
        "img":base64img,
        "count":[count]
    }
    return JsonResponse(data)