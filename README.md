To run the project, follow these steps:

Clone the project repository or download the source code.

Install the necessary dependencies. You can use the following command to install the required packages:

pip install flask torchvision pillow tensorflow
Place your model files in the appropriate locations. In the provided code, the ResNet model is expected to be located at models/Model.pth, and the Keras model files (_model_.json and _model_.h5) are expected to be in the md folder.

Make sure you have the desired HTML templates (index.html, result.html, resultAll.html, All.html, and BreastCancer.html) in the same directory as your Python script.

Run the Flask application by executing the following command in the terminal:
python your_script_name.py
Replace your_script_name.py with the actual name of your Python script.

Once the application is running, open your web browser and visit http://localhost:5000 to access the home page.

Use the provided web interface to upload and predict the class for the images using the ResNet model (/predictAll route) or the Keras model (/predict route).

That's it! You should now be able to run and use the project. Make sure to adjust the file paths and model names if necessary, and ensure that your models are properly trained and saved before running the application.




