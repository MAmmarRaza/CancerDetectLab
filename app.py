from flask import Flask, render_template, request, redirect, url_for
import torch
import torchvision.models as models
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import base64
import io
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the saved ResNet model
resnet_model = models.resnet152(num_classes=4)  # Replace with the appropriate model architecture
resnet_model_path = 'models/Model.pth'
resnet_model.load_state_dict(torch.load(resnet_model_path))
resnet_model.eval()

# Define a dictionary to map ResNet labels to class names
resnet_class_names = {
    0: 'Benign',
    1: 'Early',
    2: 'Pre',
    3: 'Pro'
    # Add more labels and class names as needed
}

# Load the saved Keras model
with open('md/_model_.json', 'r') as json_file:
    keras_model_json = json_file.read()

keras_model = model_from_json(keras_model_json)
keras_model.load_weights('md/_model_.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/leukemia')
def leukemia():
    return render_template('All.html')

@app.route('/BreastCancer')
def BreastCancer():
    return render_template('BreastCancer.html')

# Define the prediction route for the first project (ResNet model)
@app.route('/predictAll', methods=['POST'])
def predictAll():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return render_template('result.html', error='No image file provided.')

    # Read and preprocess the image for ResNet model
    image = request.files['image']
    img = Image.open(image)
    resnet_input_tensor = preprocess_image_resnet(img)

    # Make a prediction using ResNet model
    resnet_predicted_label, resnet_predicted_class_name = predict_image_resnet(resnet_input_tensor)

    # Prepare the image for display
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Calculate the size of the image data in bytes
    img_size = len(img_byte_arr.getvalue())

    # Prepare the response for ResNet model
    resnet_response = {
        'predicted_label': resnet_predicted_label,
        'predicted_class_name': resnet_predicted_class_name,
        'image_details': {
            'filename': image.filename,
            'size': img_size,
            'type': image.content_type,
            'data': img_data
        }
    }

    return render_template('resultAll.html', resnet_response=resnet_response)


# Define the prediction route for the second project (Keras model)
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return redirect(url_for('home'))

    # Read and preprocess the image for Keras model
    image = request.files['image']
    img = Image.open(image)
    img = img.resize((100, 100))
    img = np.array(img)
    img = np.true_divide(img, 255)
    img = img.reshape(-1, 100, 100, 3)
    keras_predictions = predict_image_keras(img)
    keras_predicted_class = np.argmax(keras_predictions)
    keras_classes = ['Non Invacive Ductal Carcinoma', 'Invacive Ductal Carcinoma']
    predicted_label = keras_classes[keras_predicted_class]

    # Prepare the image for display
    img_byte_arr = io.BytesIO()
    Image.fromarray((img.squeeze() * 255).astype(np.uint8)).save(img_byte_arr, format='PNG')
    img_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Calculate the size of the image data in bytes
    img_size = len(img_byte_arr.getvalue())

    # Prepare the response for Keras model
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


def preprocess_image_resnet(image):
    # Apply transformations
    transform = Compose([
        Resize(224),  # Resize the image to a desired size
        ToTensor()  # Convert the image to a PyTorch tensor
    ])

    preprocessed_image = transform(image)

    # Add a batch dimension to the preprocessed image
    input_tensor = preprocessed_image.unsqueeze(0)

    return input_tensor


def predict_image_resnet(input_tensor):
    # Make predictions with ResNet model
    with torch.no_grad():
        outputs = resnet_model(input_tensor)

    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Get the predicted class label
    predicted_label = torch.argmax(outputs, dim=1).item()

    # Look up the predicted class name based on the label
    predicted_class_name = resnet_class_names.get(predicted_label, 'Unknown')

    return predicted_label, predicted_class_name


def predict_image_keras(input_image):
    # Make predictions with Keras model
    predictions = keras_model.predict(input_image)

    return predictions


if __name__ == '__main__':
    app.run(debug=True)
