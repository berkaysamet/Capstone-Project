
import cv2
import numpy as np
import math
import argparse
from flask import Flask, render_template, Response, request
from PIL import Image
import csv
import os
import requests

PROJEDENEME2 = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PROJEDENEME2


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", default="./face_detector")
ap.add_argument("-a", "--age", default="./age_detector")
ap.add_argument("-g", "--gender", default="./gender_detector")
ap.add_argument("-c", "--conf", type=float, default=0.5)
args = vars(ap.parse_args())

def load_models(face_path, age_path, gender_path):
    print("[BILGI] Yüz dedektör modeli yükleniyor...")
    prototxtPath = os.path.sep.join([face_path, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("[BILGI] Yaş dedektör modeli yükleniyor...")
    prototxtPath = os.path.sep.join([age_path, "age_deploy.prototxt"])
    weightsPath = os.path.sep.join([age_path, "age_net.caffemodel"])
    ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("[BILGI] Cinsiyet dedektör modeli yükleniyor...")
    prototxtPath = os.path.sep.join([gender_path, "gender.prototxt"])
    weightsPath = os.path.sep.join([gender_path, "gender_net.caffemodel"])
    genderNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    return faceNet, ageNet, genderNet

faceNet, ageNet, genderNet = load_models(args["face"], args["age"], args["gender"])


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# dedektörün tahmin edeceği yaş sınıflarının listesi tanımlanır
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
	# 0=male 1=female
GENDER_BUCKETS = ["Male", "Female"]
frame_results = []
def detect_and_predict_age(frame, faceNet, ageNet, genderNet, minConf=0.5):
	
	results = []
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > minConf:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue
			faceBlob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
			global j
			j = i
			ageNet.setInput(faceBlob)
			predsA = ageNet.forward()
			i = predsA[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = predsA[0][i]
			genderNet.setInput(faceBlob)
			predsG = genderNet.forward()
			j = predsG[0].argmax()
			gender = GENDER_BUCKETS[j]
			genderConfidence = predsG[0][j]

			d = {"loc": (startX, startY, endX, endY), "age": (age, ageConfidence), "gender": (gender, genderConfidence)}
			results.append(d)

			for result in results:
				age, ageConfidence = result['age']
				gender, genderConfidence = result['gender']
				
				if ageConfidence > 0.6 and genderConfidence > 0.6:
					frame_results.append((age, gender))
				if len(frame_results) == 30:
					most_frequent_result = max(set(frame_results), key = frame_results.count)
					with open('output.csv', 'a',newline='') as f:
						writer = csv.writer(f)
						writer.writerow([most_frequent_result[0], most_frequent_result[1]])
						writer.writerow([ageConfidence])
						# f.write(f'{most_frequent_result}\n')
						# f.write(f'{ageConfidence}\n')				
					frame_results.clear()

	return results

def send_data(results,age, gender, confidence_score):
	url = 'http://127.0.0.1:8080/rest/add_result'
	data = {'age_group': age, 'gender': gender, 'confidence_score' : confidence_score,'visit_id' : 3}
	params = {'age_group': age, 'gender': gender, 'confidence_score' : confidence_score,'visit_id' : 3}
	response = requests.post(url,params=params)
	print("Status Code:", response.status_code)

	for result in results:
		age, ageConfidence = result['age']
		gender, genderConfidence = result['gender']
		
		if ageConfidence > 0.6 and genderConfidence > 0.6:
			frame_results.append((age, gender))
		if len(frame_results) == 30:
			most_frequent_result = max(set(frame_results), key = frame_results.count)
			send_data(most_frequent_result[0], most_frequent_result[1], ageConfidence)
			frame_results.clear()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        cap = cv2.VideoCapture(1)
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                else:
                    results = detect_and_predict_age(frame, faceNet, ageNet, genderNet)
                    for r in results:
                        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100) + "  " + "{}: {:.2f}%".format(r["gender"][0], r["gender"][1] * 100)
                        (startX, startY, endX, endY) = r["loc"]
                        y = startY - 20 if startY - 20 > 20 else startY + 20
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 3)
                        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (205, 205, 0), 2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
              cap.release()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

if __name__ == '__main__':
    app.run(debug=True)
