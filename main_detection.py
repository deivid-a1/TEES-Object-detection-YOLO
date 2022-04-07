from detection import Detector
import os

def main():
    videoPath = "D:/Videos/unknown/interview.mp4" #colocar

    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    base = Detector(videoPath, configPath, modelPath, classesPath)
    base.onCapture()


if __name__ == '__main__':
    main()