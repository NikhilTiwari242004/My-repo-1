from flask import Flask, request, render_template
import numpy as np
import pickle
import webbrowser
from threading import Timer
import matplotlib
matplotlib.use('Agg')   # ✅ Non-GUI backend for Flask
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model parameters
theta, X_mean, X_std = pickle.load(open("house_model.pkl", "rb"))

# Exchange rate (approx) for USD → INR
INR_RATE = 83.0  

# Prediction function
def predict_price(area, bedrooms, age):
    x = np.array([area, bedrooms, age], dtype=float)
    x_norm = (x - X_mean) / (X_std + 1e-8)  # avoid division by zero
    x_b = np.r_[1, x_norm]  # add bias term
    return x_b.dot(theta)

@app.route("/")
def home():
    return render_template("index.html", prediction_text=None, show_prediction_graph=False)

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = float(request.form["bedrooms"])
    age = float(request.form["age"])
    
    prediction_usd = predict_price(area, bedrooms, age)
    prediction_inr = prediction_usd * INR_RATE   # ✅ Convert to INR

    # --- Generate prediction graph ---
    os.makedirs("static/plots", exist_ok=True)
    plt.figure(figsize=(5,4))
    plt.bar(["Predicted Price"], [prediction_inr], color="royalblue")
    plt.ylabel("Price (INR)")
    plt.title("Predicted House Price")
    plt.tight_layout()
    plt.savefig("static/plots/prediction.png")
    plt.close()
    # --- End graph ---

    return render_template(
        "index.html",
        prediction_text=f"Estimated House Price: ₹{prediction_inr:,.2f}",
        show_prediction_graph=True
    )

# Auto-open browser once
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    from werkzeug.serving import is_running_from_reloader
    if not is_running_from_reloader():
        Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=True)
