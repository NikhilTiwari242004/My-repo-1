from flask import Flask, request, render_template
import numpy as np
import pickle
import webbrowser
from threading import Timer

app = Flask(__name__)

# Load model parameters
theta, X_mean, X_std = pickle.load(open("house_model.pkl", "rb"))

# Prediction function
def predict_price(area, bedrooms, age):
    x = np.array([area, bedrooms, age], dtype=float)
    x_norm = (x - X_mean) / X_std
    x_b = np.r_[1, x_norm]  # add bias
    return x_b.dot(theta)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = float(request.form["bedrooms"])
    age = float(request.form["age"])
    
    prediction = predict_price(area, bedrooms, age)
    return render_template("index.html", prediction_text=f"Estimated House Price: ${prediction:,.2f}")

# Auto-open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)
