from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import io
import base64
from flask import Flask, render_template, url_for
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the saved model from disk
with open('md/_model_.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights('md/_model_.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return redirect(url_for('home'))

    # Read and preprocess the image
    image = request.files['image']
    img = Image.open(image)
    img = img.resize((100, 100))
    img = np.array(img)
    img = np.true_divide(img, 255)
    img = img.reshape(-1, 100, 100, 3)

    # Make a prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Map the predicted class index to the actual class label
    classes = ['Non Invacive Ductal Carcinoma', 'Invacive Ductal Carcinoma']
    predicted_label = classes[predicted_class]

    # Prepare the image for display
    img_byte_arr = io.BytesIO()
    Image.fromarray((img.squeeze() * 255).astype(np.uint8)).save(img_byte_arr, format='PNG')
    img_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Calculate the size of the image data in bytes
    img_size = len(img_byte_arr.getvalue())
    
    # Prepare the response
    response = {
        'predicted_class': predicted_label,
        'image_details': {
            'filename': image.filename,
            'size': img_size,
            'type': image.content_type,
            'data': img_data
        }
    }

    return render_template('result.html', result=response)

if __name__ == '__main__':
    app.run(debug=True)

