from flask import Flask, render_template, jsonify, request 
import streamlit as st
import joblib
import numpy as np 

app = Flask(__name__)

# Load your machine learning model
model = joblib.load('model.pkl')  # My trained model, saved as 'model.pkl'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/streamlit')
def streamlit():
    st.set_page_config(page_title="My Streamlit App")
    st.write("Hello, world!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features'])
    prediction = model.predict(features.reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
