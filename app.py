from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/wine_cultivar_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "alcohol": float(request.form["alcohol"]),
            "malic_acid": float(request.form["malic_acid"]),
            "alcalinity_of_ash": float(request.form["alcalinity_of_ash"]),
            "magnesium": float(request.form["magnesium"]),
            "flavanoids": float(request.form["flavanoids"]),
            "color_intensity": float(request.form["color_intensity"])
        }

        df = pd.DataFrame([data])
        pred_class = model.predict(df)[0]
        prediction = f"Cultivar {pred_class + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
