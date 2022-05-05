import cv2
import numpy as np
import glob
import random
from windowcapture import WindowCapture
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")
#Uncomment if want to use GPU NVIDIA
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Name custom object
classes = ["sign_green", "sign_red", "sign_yellow"]

# Images path
capturePath = WindowCapture("Grand Theft Auto V")
#capturePath = WindowCapture("Nova guia - Brave")

# while True:
#     capturePath.get_screenshot()
#     print(capturePath.w, capturePath.h)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
startTime = 0
while(True):
    # Insert here the path of your images
    img = capturePath.get_screenshot()

    currentTime = time.time()    
    fps = 1/(currentTime - startTime)
    startTime = currentTime 

    # Loading image
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            if label == "sign_red":
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
            elif label == "sign_green":
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # if label == "sign_red":
            #     cv2.putText(img, '', (x-30, y - 3), font, 1.5, (255,255,255), 2)
            # elif label == "sign_green":
            #     cv2.putText(img, '', (x-30, y - 3), font, 1.5, (34,139,34), 2)
            # else:
            #     cv2.putText(img, '', (x-30, y - 3), font, 1.5, (255, 117, 24), 2)

            
    print(fps)
    cv2.putText(img, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break


cv2.destroyAllWindows()