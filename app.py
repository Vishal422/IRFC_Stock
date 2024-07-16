# from flask import Flask, request, jsonify
# import numpy as np
# import pandas as pd
# import pickle
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the model and scaler
# model = load_model('lstm_model.h5')
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the data from the request
#     data = request.get_json(force=True)
    
#     # Convert data into dataframe
#     df = pd.DataFrame(data, index=[0])
    
#     # Scale the data
#     scaled_data = scaler.transform(df)
    
#     # Reshape for LSTM model
#     scaled_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
    
#     # Make prediction
#     prediction = model.predict(scaled_data)
#     prediction = scaler.inverse_transform(prediction)
    
#     # Return the prediction
#     return jsonify({'prediction': prediction[0, 0]})

# if __name__ == '__main__':
#     app.run(debug=True,  host='0.0.0.0', port=5000)




from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model and scaler
with open('lstm_model.h5', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Stock Price Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['Open'], data['High'], data['Low'], data['Volume']]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
