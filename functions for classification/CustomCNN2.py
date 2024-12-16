import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

# architecture
model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')  # 5 classes
])

model2.load_weights('saved_model2.h5')

# class names
classes = ['Drones', 'Helicopters', 'UAVs', 'Fighterjets', 'Airplanes']


def predict_CNN_2(image):

    try:
        # Ensure the image is resized to 128x128
        img_width, img_height = 128, 128
        resized_image = cv2.resize(image, (img_width, img_height))

        # Normalize pixel values
        resized_image = resized_image / 255.0

        resized_image = np.expand_dims(resized_image, axis=0)

        # Predict with the model
        predictions = model2.predict(resized_image)

        # Get index of the class with the highest probability
        predicted_class_idx = np.argmax(predictions, axis=1)[0]

        # Map index to class name
        predicted_class = classes[predicted_class_idx]

        return predicted_class

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
