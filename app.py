from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model with absolute path
model = load_model(r'C:\Users\DELL\Desktop\project_for_final_year\fruit_quality_detector\3rdVersion\models\apple_quality_model.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize to match your model's input shape
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    quality = None  # Initialize quality variable
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the uploaded image
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)

            # Preprocess and predict
            img_array = preprocess_image(image_path)
            prediction = model.predict(img_array)
            quality = 'Good' if prediction[0][0] > 0.5 else 'Bad'

            # Optionally, remove the uploaded image after processing
            os.remove(image_path)

    return render_template('index.html', quality=quality)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create uploads directory if it doesn't exist
    app.run(debug=True)
