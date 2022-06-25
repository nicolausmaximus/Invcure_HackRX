import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import re
from pytesseract import Output

def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def fetchText(file_name):
    # file_name = "S_2.png"
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) 
    plot_gray(image)

    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])
    boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
        boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # plot_rgb(boxes)
    extracted_text = pytesseract.image_to_string(image)
    print(extracted_text)

fetchText('S_2.png')