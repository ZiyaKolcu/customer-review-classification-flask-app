from flask import Flask, request, render_template, jsonify
from huggingface_hub import hf_hub_download
import joblib

app = Flask(__name__)

model_filenames = {
    "lr_model": "LogisticRegression_model.joblib",
    "gb_model": "GradientBoostingClassifier_model.joblib",
    "sgd_model": "SGDClassifier_model.joblib",
    "preprocessor": "preprocessor.joblib",
}

models = {}

for name, filename in model_filenames.items():
    path = hf_hub_download(
        repo_id="ZiyaKolcu/CustomerReviewClassification", filename=filename
    )
    models[name] = joblib.load(path)

lr_model = models["lr_model"]
gb_model = models["gb_model"]
sgd_model = models["sgd_model"]
preprocessor = models["preprocessor"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.get("input_value")
    selected_model = request.form.get("model_name")

    if not data or selected_model not in ["model-1", "model-2", "model-3"]:
        return jsonify({"error": "Invalid input"}), 400

    if isinstance(data, str):
        data = [data]

    processed_text = preprocessor.transform(data)

    if selected_model == "model-1":
        prediction = lr_model.predict(processed_text)
    elif selected_model == "model-2":
        prediction = gb_model.predict(processed_text)
    elif selected_model == "model-3":
        prediction = sgd_model.predict(processed_text)
    else:
        return jsonify({"error": "Invalid model selected"}), 400

    prediction_label = "positive" if int(prediction[0]) == 1 else "negative"

    return jsonify({"prediction": prediction_label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
