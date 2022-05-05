import cv2 as cv
import numpy as np
import time
from windowcapture import WindowCapture
np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.base = cv.dnn_DetectionModel(self.modelPath, self.configPath)
        self.base.setInputSize(320,320)
        self.base.setInputScale(1.0/127.5)
        self.base.setInputMean((127.5, 127.5, 127.5))
        self.base.setInputSwapRB(True)

        self.classRead()

    def classRead(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        #self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0,  high=255, size=(len(self.classesList), 3))
        print(self.classesList)

    def onCapture(self):

        #cap = cv.VideoCapture(self.videoPath)

        
        #cap = WindowCapture("Grand Theft Auto V")
        cap = WindowCapture("Nova guia - Brave")
        #cap = WindowCapture("Grand Theft Auto V")



        

        startTime = 0
        while True:

            image = cap.get_screenshot()
            currentTime = time.time()

            fps = 1/(currentTime - startTime)

            startTime = currentTime

            classLabelIDs, confidences, bboxs, = self.base.detect(image, confThreshold = 0.5)
            bbox = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)

            if (len(bboxIdx) != 0):
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x,y, w,h = bbox

                    cv.rectangle(image, (x,y), (x+w, y+h), color=classColor, thickness=1)
                    cv.putText(image, displayText, (x,y-10), cv.FONT_HERSHEY_PLAIN, 1, classColor, 2)
            cv.putText(image, "FPS: " + str(int(fps)), (20,70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv.imshow("Result", image)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
        cv.destroyAllWindows()
