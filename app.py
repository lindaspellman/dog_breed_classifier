# pip install Flask # how to install Flask 

# how to set up a virtual environment
# python -m venv venv
# source venv/Scripts/activate # this did not work, but I got a popup asking about using the new environment for the current workspace 

# suddenly, out of now where these imports got yellow squiggly lines under them. 
from flask import Flask, jsonify, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your machine learning model
model = joblib.load('model.pkl')  # My trained model, saved as 'model.pkl'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features'])
    prediction = model.predict(features.reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)