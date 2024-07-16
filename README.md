## Stock Price Prediction using Machine Learning and Flask

This project involves building a machine learning model to predict stock prices and deploying it as a web application using Flask, hosted on an AWS EC2 instance.

 Project Overview

1. Dataset:
   - Uses historical stock price data of IRFC.
   - Includes features like Open, High, Low, and Volume, with Close as the target variable.

2. Model Development:
   - Preprocessed data to handle missing values and scale features.
   - Split data into training and testing sets.
   - Trained a RandomForestRegressor model.
   - Evaluated model performance using RMSE and MAE.

3. Flask Application:
   - Created a Flask API to serve the model.
   - The `/predict` endpoint accepts JSON input with stock features and returns the predicted closing price.

4. Deployment:
   - Deployed the Flask application on an AWS EC2 instance.
   - Configured security groups to allow HTTP and SSH access.
   - Hosted the model and scaler files for inference.

 Repository Structure


.
├── app.py                 # Flask application
├── model.pkl              # Trained machine learning model
├── scaler.pkl             # Scaler used for feature scaling
├── requirements.txt       # List of dependencies
├── README.md              # Project description and setup instructions
└── IRFC.NS_stock_data.csv # Dataset


 How to Run

1. Clone the Repository:
   sh
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   

2. Install Dependencies:
   sh
   pip install -r requirements.txt
   

3. Run the Flask App:
   sh
   python app.py
   

4. Send a Prediction Request:
   sh
   curl -X POST http://<EC2-Instance-IP>:5000/predict -H "Content-Type: application/json" -d '{"Open": 100, "High": 110, "Low": 90, "Volume": 1000000}'
   

 Contact

For any inquiries, please contact wagarajvishal@gmail.com


