from detection import Detector
import os

def main():
    videoPath = "C:/Users/deivi/Desktop/teste.png" #colocar

    configPath = os.path.join("", "yolov3_custom.cfg")
    modelPath = os.path.join("", "yolov3_custom_last.weights")
    classesPath = os.path.join("model_data", "custom.names")

    base = Detector(videoPath, configPath, modelPath, classesPath)
    base.onCapture()

if __name__ == '__main__':
    main()