import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import sys
import os
import glob
import re
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import cv2
from model import get_model

app = Flask(__name__, template_folder='templates')

def model_predict(img_path):

	model = load_model('model/model2.h5') 
	model._make_predict_function()          # Necessary 
	print('Model loaded. Check http://127.0.0.1:5000/')

	IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = 350, 350, 1
	img = Image.open(img_path)
	img = img.resize((IMG_WIDTH, IMG_HEIGHT))
	img = img.convert('L')
	im = np.asarray(img)
	im = im/255
	im_input = im.reshape((1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))

	preds = model.predict(im_input)[0]
	keras.backend.clear_session()

	return preds

def bmi_predict(img_path):
	weights_file = 'model/bmi_model_weights.h5'
	model = get_model(ignore_age_weights=True)
	model.load_weights(weights_file)

	img = cv2.imread(img_path)
	if img.shape[0]>img.shape[1]:
		halfminlength = int(img.shape[1]/2)
		mid = int(img.shape[0]/2)
		input_img = img[mid-halfminlength:mid+halfminlength,:,:]
	elif img.shape[1]>img.shape[0]:
		halfminlength = int(img.shape[0]/2)
		mid = int(img.shape[1]/2)
		input_img = img[:,mid-halfminlength:mid+halfminlength,:]
	else:
		input_img = img
	input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	input_img = cv2.resize(img, (224, 224))/255.
	
	predictions = model.predict(input_img.reshape(1, 224, 224, 3))
	label = str(predictions[0]-predictions/5)
	return label

@app.route('/', methods=['GET'])
def index():
	# Main page
	return render_template('index.html')
	
@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make prediction
		preds = model_predict(file_path)

		facevalue = {'negative': 0, 'positive': 1, 'neutral': 2}
		# Process your result for human
		# pred_class = preds.argmax(axis=-1)
		pred_class = list(facevalue.keys())[preds.argmax()]        

		bmi = bmi_predict(file_path)[2:]

		result = 'This person is showing ' + str(pred_class) + ' emotion.'      
		result2 = 'The predicted BMI is ' + str(bmi)       
		return result + result2
	return None

if __name__ == '__main__':
	app.run(debug=True)