import cv2
import cvzone
from PIL import Image
from classification15ep import classify_image15
from classification8ep import classify_image8
from classification10ep import classify_image10
from CustomCNN1 import predict_CNN
from CustomCNN2 import predict_CNN_2
from voting import majority_voting
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from classificationvgg16 import classify_vgg

# Directory containing test data
test_data_dir = r"C:\Users\KHALED\OneDrive - Rafic Hariri University\Desktop\classes test"  # Update this path as needed

# Classes (subfolder names)
classes = ["Airplanes", "Helicopters", "Drones", "Fighterjets", "UAVs"]

# Store true labels, final predictions, and durations
true_values = []
final_predictions = []
durations = []  # List to store prediction durations

# Process each image in the subfolders
for class_name in classes:
    class_dir = os.path.join(test_data_dir, class_name)
    for image_file in os.listdir(class_dir):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(class_dir, image_file)
            img = cv2.imread(img_path)

            # Measure start time
            start_time = time.time()

            # True label from folder name
            true_values.append(class_name)

            # Convert to PIL Image
            pil_image = Image.fromarray(img)

            # Perform classifications
            #subclass15 = classify_image15(pil_image)
            #subclass8 = classify_image8(pil_image)
            subclass10 = classify_image10(pil_image)
            #subclasscnn1 = predict_CNN(img)
            subclasscnn2 = predict_CNN_2(img)
            subclassvvg = classify_vgg(img)

            # Perform majority voting
            predictions = [subclass10, subclassvvg, subclasscnn2]
            #predictions = [subclass10, subclass8, subclass15, subclasscnn1, subclasscnn2]

            final_prediction = majority_voting(predictions)

            # Store the final prediction
            final_predictions.append(final_prediction)

            # Measure end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            durations.append(duration)

# Calculate average prediction time
average_time = sum(durations) / len(durations)

print("Processing complete. Generating confusion matrix and accuracy score for final predictions...")

# Function to save confusion matrix as an image
def save_confusion_matrix(cm, model_name, class_names, save_path, accuracy, avg_time):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(
        f'Confusion Matrix for {model_name}\nAccuracy: {accuracy * 100:.2f}% | Avg Time: {avg_time:.2f} sec',
        fontsize=14
    )
    plt.colorbar()

    # Set axis ticks to the class names
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    # Annotate each cell with its corresponding value
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Generate confusion matrix and calculate accuracy for the final predictions
print("\nEvaluating Majority Voting System...")
# Confusion matrix
cm = confusion_matrix(true_values, final_predictions, labels=classes)
# Accuracy
accuracy = accuracy_score(true_values, final_predictions)
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Prediction Time: {average_time:.2f} seconds per image")

# Save confusion matrix as an image
save_path = r"C:\Users\KHALED\Downloads\final_prediction_confusion_matrix.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
save_confusion_matrix(cm, "Majority Voting System", classes, save_path, accuracy, average_time)
print(f"Saved confusion matrix image to {save_path}")
