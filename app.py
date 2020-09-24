from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import os, fnmatch
# Keras
import librosa
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
INPUT_DIR = "C:/Users/tanma/Desktop/Data Science/BE/inpu/"
SAMPLE_RATE = 16000
# seconds
MAX_SOUND_CLIP_DURATION=12  
# Define a flask app
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Map label text to integer
CLASSES = ['artifact','murmur','normal']
# {'artifact': 0, 'murmur': 1, 'normal': 3}
NB_CLASSES=len(CLASSES)

# Map integer value to text labels
label_to_int = {k:v for v,k in enumerate(CLASSES)}
print (label_to_int)
print (" ")
# map integer to label text
int_to_label = {v:k for k,v in label_to_int.items()}



app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH ='best_model_trained.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    
    duration=12
    sr=16000
    data = []
    input_length=sr*duration 
    X, sr = librosa.load(img_path, sr=sr, duration=duration,res_type='kaiser_fast') 
    dur = librosa.get_duration(y=X, sr=sr)
    # pad audio file same duration
    if (round(dur) < duration):
        print ("fixing audio lenght :")
        y = librosa.util.fix_length(X, input_length)                
    #normalized raw audio 
    # y = audio_norm(y)            
    # extract normalized mfcc feature from data
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)      
    feature = np.array(mfccs).reshape([-1,1])
    data.append(feature)
    
    A_test = np.asarray(data,dtype = None, order = None)
    
    y_pred = model.predict_classes(A_test, batch_size=32)
    print ("The patient's heart sound is "+ int_to_label[y_pred[0]])
    
    
    
    preds=y_pred[0]
    if preds==0:
        preds="artifact"
    elif preds==1:
        preds="murmur"
    else:
        preds="normal"
    
    
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
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False)