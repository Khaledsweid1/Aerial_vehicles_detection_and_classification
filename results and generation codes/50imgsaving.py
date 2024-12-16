import cv2
import cvzone
from PIL import Image
from classification15ep import classify_image15
from classification8ep import classify_image8
from classification10ep import classify_image10
from CustomCNN1 import predict_CNN
from CustomCNN2 import predict_CNN_2
from voting import majority_voting
import time
import os
from classificationvgg16 import classify_vgg


# Directory containing images
image_dir = r"C:\Users\KHALED\OneDrive - Rafic Hariri University\Desktop\testing folder"
output_dir = r"C:\Users\KHALED\OneDrive - Rafic Hariri University\Desktop\testing folder_output"
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
        img_path = os.path.join(image_dir, image_file)
        img = cv2.imread(img_path)

        # Measure operation duration
        start_time = time.time()

        pil_image = Image.fromarray(img)
        # Perform classifications
        #subclass15 = classify_image15(pil_image)
        #subclass8 = classify_image8(pil_image)
        subclass10 = classify_image10(pil_image)
        #subclasscnn1 = predict_CNN(img)
        subclasscnn2 = predict_CNN_2(img)
        subclassvvg = classify_vgg(img)
        #predictions = [subclass10, subclass8, subclass15, subclasscnn1, subclasscnn2,subclassvvg]
        predictions = [subclass10, subclasscnn2, subclassvvg]
        final_prediction = majority_voting(predictions)

        # Display predictions
        #cvzone.putTextRect(img, f'{subclass15}', (20, 20), scale=1, thickness=1, colorR=(0, 255, 0))
        #cvzone.putTextRect(img, f'{subclass8}', (20, 40), scale=1, thickness=1, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{subclass10}', (20, 60), scale=1, thickness=1, colorR=(0, 0, 255))
        #cvzone.putTextRect(img, f'{subclasscnn1}', (20, 80), scale=1, thickness=1, colorR=(0, 155, 255))
        cvzone.putTextRect(img, f'{subclasscnn2}', (20, 100), scale=1, thickness=1, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{subclassvvg}', (20, 120), scale=1, thickness=1, colorR=(100, 46, 255))
        cvzone.putTextRect(img, f'{final_prediction}', (20, 140), scale=1, thickness=3, colorR=(0, 0, 0))

        # Measure operation duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"Processed {image_file} in {duration:.2f} seconds.")

        # Save processed image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, img)

print("Processing complete.")
