
import cv2
import numpy as np
import math
import argparse
from flask import Flask, render_template, Response, request
from PIL import Image
import io

PROJEDENEME2 = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PROJEDENEME2


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []

    for i in range(detections.shape[2]): 
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:   
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()
data = []


def gen_frames():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)','(48-53)', '(60-100)']
    genderList = ['Male', 'Female']


    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(1)
    padding = 20
    while cv2.waitKey(1) < 0:
        
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

  
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:   
            print("No face detected")   

        for faceBox in faceBoxes:
            
            face = frame[max(0, faceBox[1]-padding):  
                         min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
        
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]


            ageNet.setInput(blob)
        
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]


                    # Append the new data to the list
            data.append((gender, age))

            # Remove the oldest data if the list has more than 60 items
            if len(data) > 60:
                data.pop(0)

            # Write the most frequent age and gender to a text file
            if len(data) == 60:
                most_frequent = max(set(data), key=data.count)
                with open('output.txt', 'a') as f:
                    f.write(f'Most accurate gender: {most_frequent[0]}, Most accurate age: {most_frequent[1][1:-1]} years\n')


      
            cv2.putText(resultImg, f'{gender}, {age}', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        #cv2.imshow("Detecting age and gender", resultImg)

            if resultImg is None:
                continue

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            #resultImg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


def gen_frames_photo(img_file):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

 
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)



    frame = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    frame = img_file
    hasFrame, frame = img_file.read()
    ret, frame = cv2.imencode('.jpg', img_file)
    video = cv2.VideoCapture(img_file)
    padding = 20
    while cv2.waitKey(1) < 0:
        
         hasFrame, frame = video.read()
         if not hasFrame:
           cv2.waitKey()
         break

       
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:   
            print("No face detected") 

    for faceBox in faceBoxes:
            
            face = frame[max(0, faceBox[1]-padding):   
                         min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]


            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')  

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years') 

            

            cv2.putText(resultImg, f'{gender}, {age}', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)

            if resultImg is None:
                continue

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            #resultImg = buffer.tobytes()
            return (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['fileToUpload'].read()
        img = Image.open(io.BytesIO(f))
        img_ip = np.asarray(img, dtype="uint8")
        print(img_ip)
        return Response(gen_frames_photo(img_ip), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
