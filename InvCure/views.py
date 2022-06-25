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
    fetchText('S_2.png')
    return Response({"message":"success"})