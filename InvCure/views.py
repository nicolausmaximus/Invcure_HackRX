from select import select
from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail
from InvCure import settings
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth import authenticate, login, logout
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
from database import *
from plainRecognition import *
from ner import *
from metadata import *
import environ
env = environ.Env()
environ.Env.read_env()

def index(request):
    return render(request, 'index.html')

def authentication(request):
    return render(request, 'login_signup.html')

def upload(request):
    return render(request, 'upload.html')

@api_view(['POST'])
def parseImage(request):
<<<<<<< HEAD
    # extracted_text= fetchText('S_2.png')
    extracted_text= fetchText('S_1.jpg')
    # print(extracted_text)
    NamedER(extracted_text)
    # openaiNERPatientName(extracted_text)
    # openaiNERAddress(extracted_text)
    # openaiNERPhoneNumber(extracted_text)
    # openaiNEREmail(extracted_text)
    # openaiNERDate(extracted_text)
    # openaiNERGender(extracted_text)
    # openaiNERAmount(extracted_text)
    # openaiNERItems(extracted_text)
    extractAll(extracted_text)
    return Response({"message":"success"})
=======
    if(metadatacheck('S_2.png')):
        extracted_text= fetchText('S_2.png')
        print(extracted_text)
        NamedER(extracted_text)
        return Response({"message":"success"})
    else:
        print("edited")
        return render(request, 'edited.html')
>>>>>>> 19a33bd (rebase)
