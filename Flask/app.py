import pickle
import numpy as np
from flask import Flask, render_template, request


model = pickle.load(open("rf_acc_100.pkl", "rb"))
normalizer = pickle.load(open("normalizer.pkl", "rb"))


input_features = [
    'Age', 'Gender', 'Total Bilirubin    (mg/dl)', 'Direct    (mg/dl)',
    'AL.Phosphatase      (U/L)', 'SGPT/ALT (U/L)', 'SGOT/AST      (U/L)',
    'Total Protein     (g/dl)', 'Albumin   (g/dl)', 'A/G Ratio'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form.get(f)) for f in input_features]
        input_array = np.array(input_data).reshape(1, -1)
        input_normalized = normalizer.transform(input_array)


        prediction = model.predict(input_normalized)[0]
        probability = model.predict_proba(input_normalized)[0][int(prediction)]


        label = "🟢 Negative for Liver Cirrhosis" if prediction == 0 else "🔴 Positive for Liver Cirrhosis"
        tip = (
            "Your liver health indicators seem normal. Maintain a healthy diet and regular checkups." if prediction == 0
            else "Your inputs show risk markers. We recommend you consult a hepatologist for further diagnosis."
        )

        return render_template(
            'index.html',
            prediction_text=label,
            confidence=f"{probability * 100:.2f}%",
            health_tip=tip
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
