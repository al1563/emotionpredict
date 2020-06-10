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
		result = 'This person is showing ' + str(pred_class) + ' emotion.'              
		return result
	return None

if __name__ == '__main__':
	app.run(debug=True)