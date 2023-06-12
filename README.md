# Cancer Detection Project

This project implements a web application for cancer detection using deep learning models. It consists of two separate projects: one using a ResNet model for detecting various types of cancer, and another using a Keras model specifically for breast cancer detection.

## Features

- Predict the presence of cancer in medical images.
- Supports two different deep learning models for cancer detection.
- Provides a user-friendly web interface for uploading and predicting images.
- Displays the predicted class and image details.

## Prerequisites

Before running this project, make sure you have the following dependencies installed:

- Python 3.x
- Flask
- TorchVision
- Pillow
- TensorFlow
- NumPy


## Usage

1. Clone this repository to your local machine or download the source code.

2. Navigate to the project directory.

3. Place your trained model files in the appropriate locations:
   - For the ResNet model, place the model file in the `models` directory.
   - ResNet Model is not present in Repository you can download it from the link:
   - For the Keras model, place the model JSON file and weight file in the `md` directory.

4. Open a terminal or command prompt and run the following command to start the Flask application:
-python app.py


5. Open your web browser and visit `http://localhost:5000` to access the web application.

6. Use the provided interface to upload an image and get predictions for cancer detection.

## Folder Structure

The folder structure of this project is as follows:

- `models/`: Contains the saved model file for the ResNet model.
- `md/`: Contains the saved model JSON file and weight file for the Keras model.
- `templates/`: Contains the HTML templates for the web application.
- `static/`: Contains static files such as CSS stylesheets and images.
- `app.py`: The main Flask application script.
- `README.md`: This file.

## License

Feel free to modify and adapt this project to suit your needs.



