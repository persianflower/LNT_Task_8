
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow import keras, float16
from PIL import Image
app = Flask(__name__)
import cv2

with open('fashion_model.pkl','rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Fashion data prediction app working!"

@app.route('/predict_json',methods=['POST'])
def predict_json():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error":"No data provided"}),400
        data = np.array(data, dtype=np.float32)
        if data.size != 28 * 28:
            return jsonify({
                "error": f"Expected 784 pixels, got {data.size}"
            }), 400
        data = data.reshape(1, 28, 28, 1)
        data = data / 255.0
        prediction = model.predict(data)
        predicted_class = int(np.argmax(prediction[0]))
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error":str(e)}),400

if __name__=="__main__":
    app.run(debug=True,port=5000)


