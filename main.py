from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
import random
import requests
import psycopg2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

def detection(img_path,img_name):
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
    classes = ["numberplate"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    img = cv2.imread(img_path)
    roi = img.copy()
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    detected,output_filename,dest=None,None,None

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    if len(boxes)==0:
        print("Number plate not detected. Please click another photo by changing camera position and the angle.")
        detected=False
    else:
        detected=True
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
        plate=roi[y:y+h, x:x+w, :]
        basepath = os.path.dirname(__file__)
        output_filename='output_'+img_name
        dest=os.path.join(basepath,"uploads\output",output_filename)
        cv2.imwrite(dest,img)
        cropped_output_filename='Cropped_output_'+img_name
        dest=os.path.join(basepath,"uploads\output",cropped_output_filename)
        cv2.imwrite(dest,plate)
    return detected,output_filename,dest


def ocr_space_file(filename, overlay=False, api_key='helloworld', language='eng'):
    payload = {'isOverlayRequired': overlay,
               'apikey': '4b8e6659de88957',
               'language': language,
               'filetype':'JPG',
               'scale':'true',
               'OCREngine':2
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',files={filename: f},data=payload,)
    return r.json()

def check_database(number):
    try:
        connection = psycopg2.connect(user = "postgres", password = "vikesh12345",host = "localhost",port = "5432",database = "Capstone_Project")
        cursor = connection.cursor()
        select_query = "select * from car where car_registration_no = %s"

        cursor.execute(select_query, (number,))
        record = cursor.fetchone()
        cursor.close()
        connection.close()
        return record

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
        return null

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        detected,output_filename,plate_file_path=detection(file_path,f.filename)
        if detected==True:
            response = ocr_space_file(filename=plate_file_path, language='eng')
            number=response['ParsedResults'][0]['ParsedText'].replace(" ", "")
            number=re.sub(r'[^A-Za-z0-9]+', '', number)
            car_details=check_database(number)
            return render_template('index.html',detected=True,output=car_details,img_loc=output_filename,output_number=number)
        else:
            return render_template('index.html',detected=False,output=None,img_loc=f.filename,output_number=None)
    return None

@app.route('/output/<filename>')
def output_image(filename):
	return send_from_directory('uploads/output',filename)

@app.route('/input/<filename>')
def input_image(filename):
    return send_from_directory('uploads',filename)

if __name__ == '__main__':
    app.run(debug=True)
