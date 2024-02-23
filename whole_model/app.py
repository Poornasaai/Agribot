from flask import Flask,render_template,redirect,url_for,flash
from flask import request
import os
import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

def return_class(img_path):
    model=load_model('model.keras')
    img = image.load_img(img_path, target_size=(512, 512))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    # Depending on how you trained your model, you may need to perform additional preprocessing here
    # For example, you may need to normalize pixel values to be between 0 and 1, or scale them to the range expected by the model

    # Make predictions
    predictions = model.predict(img_array)
    print(predictions)
    # Assuming 'predictions' contains the predicted probabilities for each class
    # You can extract the class with the highest probability as the predicted class
    if(predictions[0][0]>predictions[0][1]):
        predicted_class = 0
    else:
        predicted_class = 1

    if predicted_class == 0:
        return "Rice"
    elif predicted_class == 1:
        return "Rice"
    
"""
from twilio.rest import Client
 # Your Twilio Account SID and Auth Token
account_sid = 'AC9c4da5ec7502ef1b5f12debc80622960'
auth_token = '870716b96dbfb4071ff4105b761cda64'
client = Client(account_sid, auth_token)

def send_sms(to, body):
    message = client.messages.create(
        body=body,
        from_='+15169798339',
        to=to
    )
print("SMS sent successfully!")

# Example usage:
send_sms('+91 6379656039', 'Thank you for using our service!')
"""
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/result',methods=['POST']) 
def result():
    if request.method == 'POST':
        input_image=request.files['input_image']
        print(input_image.filename)
        extension=input_image.filename.split('.')[-1]
        filename=input_image.filename.split('.')[0]
        input_image.save('static/'+input_image.filename)
        output=return_class('static/'+input_image.filename)
        return render_template("index.html",img_path=input_image.filename,output=output)
    

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    