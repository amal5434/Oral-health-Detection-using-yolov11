from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from detector import OralDetector
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialize detector
# Note: User should place best.pt in root. Using yolo11n.pt if not found.
model_file = 'best.pt' if os.path.exists('best.pt') else 'yolo11n.pt'
detector = OralDetector(model_file)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['RESULT_FOLDER'], 'detected_' + filename)
        
        file.save(input_path)
        
        # Run detection
        stats = detector.predict(input_path, output_path)
        
        return render_template('index.html', 
                               result_image='results/detected_' + filename,
                               stats=stats)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
