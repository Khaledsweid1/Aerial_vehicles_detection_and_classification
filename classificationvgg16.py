import tensorflow as tf
import cv2  # OpenCV for image processing
import numpy as np
import os

# Define the class names
CLASS_NAMES = ['Airplanes', 'Drones', 'Fighterjets', 'Helicopters', 'UAVs']


def classify_vgg(image_array):
    """
    Classify an input image (NumPy array) using the pre-trained VGG16 model in the same directory.

    Parameters:
        image_array (np.ndarray): Input image as a NumPy array.

    Returns:
        str: Predicted class name.
    """
    # Load the model from the current directory
    model_name = "vgg16_model.h5"
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Could not find {model_name} in the current directory.")
    model = tf.keras.models.load_model(model_name)

    # Resize image to 224x224
    image = cv2.resize(image_array, (224, 224))

    # Normalize pixel values
    image = image / 255.0

    # Expand dimensions to fit model's input
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_class_idx]

    # Return predicted class
    return predicted_class
