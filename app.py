from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np 

def predict_tf(image_path , loaded_model , class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    test_image = tf.keras.preprocessing.image.load_img(image_path , target_size= (256,256))  
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)    
    test_image = np.expand_dims(test_image , axis = 0)
    pred_value = loaded_model.predict(test_image)
    return class_names[np.argmax(pred_value)]


def model_load():
    loaded_model = tf.keras.models.load_model('potato_class_95.h5')
    return loaded_model

model = model_load()

#initialize the app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST']) 
def predict():
    if request.method == 'POST':
        file = request.files['file']
        print(file.filename)
        file.save(file.filename)

        data = predict_tf(file.filename, model)

        return render_template('predict.html' , value = data)
        

if __name__ == '__main__':
    app.run(debug = True)