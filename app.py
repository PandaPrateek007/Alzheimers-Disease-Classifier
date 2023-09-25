from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from skimage.io import imread, imshow
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)
model = load_model('cvalzheimers.h5')
target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['JPG' ,'jpg', 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    
    img = cv2.imread(filename)
    img = cv2.resize(img,(150,150))
    img_array = np.array(img)
    img_array = img_array.reshape(1,150,150,3)
    return img_array




@app.route('/predict',methods=['GET','POST'])
def predict():
      
      ref={0: 'MildDemented',
      1: 'ModerateDemented',
      2: 'NonDemented',
      3: 'VeryMildDemented',
     }
      
      if request.method == 'POST':
          file = request.files['file']
          
          if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)

                img = read_image(file_path)

                # Make the prediction
                answer = model.predict(img)

                # Get the class label with the highest probability
                d = ref[np.argmax(answer)]

                # Calculate the probability of the class label
                probability = round(np.max(answer) * 100, 2)

                return render_template('index.html', disease=d, prob=probability, user_image=file_path)

              
          else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)