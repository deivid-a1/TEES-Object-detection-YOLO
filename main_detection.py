from detection import Detector
import os

def main():
    videoPath = "C:/Users/deivi/Desktop/teste.png" #colocar

    configPath = os.path.join("model_data", "yolov3_testing.cfg")
    modelPath = os.path.join("model_data", "signs_training_last.weights")
    classesPath = os.path.join("model_data", "nomes.names")

    base = Detector(videoPath, configPath, modelPath, classesPath)
    base.onCapture()

if __name__ == '__main__':
    main()