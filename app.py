from flask import Flask, request, render_template
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_KEY = "YOUR_ROBOFLOW_API_KEY"
MODEL_ID = "dental-7yegp-u1pwt/1"

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # Run inference
            result = CLIENT.infer(image_path, model_id=MODEL_ID)

            # Decision logic
            if len(result["predictions"]) > 0:
                message = "⚠️ Cavity detected. Please consult a dentist."
            else:
                message = "✅ No cavity detected."

    return render_template("index.html", message=message, image=image_path)


if __name__ == "__main__":
    app.run(debug=True)

