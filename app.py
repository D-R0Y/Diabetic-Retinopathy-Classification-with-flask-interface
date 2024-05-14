from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('fusionnet_model.h5')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array=img_array/255
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict([img_array, img_array])
    predicted_class = np.argmax(predictions[0])
    
    class_labels = ['Mild', 'Moderate', 'No DR', 'Proliferate', 'severe']
    classification = class_labels[predicted_class]

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
