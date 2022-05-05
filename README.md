# TEES-Object-detection-YOLO

📜 Object detection in real time using YOLO with OpenCV. Darknet to train an IA and use their weights to detect traffic lights with screenshots from GTAV. The main target is prove that using a game like GTA V, we can detect a object from the real world. The benefit is that we can train an IA from our desktop computer, to detect objects that is so difficult to take screenshot or photos to do the training of classes (in our case sign_red, sign_green and sign_yellow).


## 🛠 Installing

install all the dependencies, in terminal (make sure that you are in repository path):

```sh
pip install -r requirements.txt
```
Now you need a .weights trained, if dont have it and want to detect traffic lights [here](https://drive.google.com/file/d/1-3PVnOn8HLxleqeagilMl-o5BLmyh7eq/view?usp=sharing), and put this in repository too (read about it in OpenCV documentation if you do not know how to use your .weights)

## 📈 Exemplo de uso

### Here you can see we taking screeshots to train our classes
![screenshot](https://github.com/deivid-a1/TEES-Object-detection-YOLO/blob/main/imgEx/ex1.png)
![classes_identify](https://github.com/deivid-a1/TEES-Object-detection-YOLO/blob/main/imgEx/ex12.PNG)

### Detection imagens from real world
![detect1](https://github.com/deivid-a1/TEES-Object-detection-YOLO/blob/main/imgEx/detect1.png)
![detect2](https://github.com/deivid-a1/TEES-Object-detection-YOLO/blob/main/imgEx/detect2.png)
![detect3](https://github.com/deivid-a1/TEES-Object-detection-YOLO/blob/main/imgEx/detect3.png)
