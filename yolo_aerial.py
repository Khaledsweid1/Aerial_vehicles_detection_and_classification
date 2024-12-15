from ultralytics import YOLO
import cv2
import cvzone
import math
from PIL import Image
from classification15ep import classify_image15
from classification8ep import classify_image8
from classification10ep import classify_image10
from CustomCNN1 import predict_CNN
from CustomCNN2 import predict_CNN_2
from classificationvgg16 import classify_vgg
from voting import majority_voting
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
model = YOLO('yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:

    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class Name
            cls = int(box.cls[0])
            if cls == 4:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                cropped_image = img[y1:y2, x1:x2]
                pil_image = Image.fromarray(cropped_image)
                #subclass15 = classify_image15(pil_image)
                #subclass8 = classify_image8(pil_image)
                subclass10 = classify_image10(pil_image)
                #subclasscnn1 = predict_CNN(cropped_image)
                subclasscnn2 = predict_CNN_2(cropped_image)
                subclassvvg = classify_vgg(cropped_image)
                #predictions = [subclass10, subclass8, subclass15, subclasscnn1, subclasscnn2,subclassvvg]
                predictions = [subclass10, subclasscnn2,subclassvvg]

                final_prediction = majority_voting(predictions)

                ##cvzone.putTextRect(img, f'{subclass15} ', (max(0, x1)+20, max(35, y1)+20), scale=1, thickness=1, colorR=(0, 255, 0))
                ##cvzone.putTextRect(img, f'{subclass8} ', (max(0, x1)+20, max(35, y1)+40), scale=1, thickness=1, colorR=(255, 0, 0))
                #cvzone.putTextRect(img, f'{subclass10} ', (max(0, x1)+20, max(35, y1)+60), scale=1, thickness=1, colorR=(0, 0, 255))
                ##cvzone.putTextRect(img, f'{subclasscnn1} ', (max(0, x1)+20, max(35, y1)+80), scale=1, thickness=1, colorR=(0, 255, 255))
                #cvzone.putTextRect(img, f'{subclasscnn2} ', (max(0, x1)+20, max(35, y1)+100), scale=1, thickness=1, colorR=(255, 0, 255))
                #cvzone.putTextRect(img, f'{subclassvvg} ', (max(0, x1)+20, max(35, y1)+120), scale=1, thickness=1, colorR=(100, 46, 255))
                cvzone.putTextRect(img, f'{final_prediction} ', (max(0, x1)+20, max(35, y2)-20), scale=1, thickness=2, colorR=(77, 46, 155))


    cv2.imshow("Image", img)
    cv2.waitKey(1)