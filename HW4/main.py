import sys
import UI
import cv2
import numpy as np
from PyQt5 import QtWidgets


class Window(QtWidgets.QWidget, UI.Ui_Form):
    
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button connect function
        self.LoadVideo.clicked.connect(self.FunctionLoadVideo)
        self.LoadImage.clicked.connect(self.FunctionLoadImage)
        self.LoadFolder.clicked.connect(self.FunctionLoadFolder)
        self.BS.clicked.connect(self.FunctionBS)
        self.Preprocessing.clicked.connect(self.FunctionPreprocessing)

    def FunctionLoadVideo(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.video = cv2.VideoCapture(path[0])
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.label_vedio.setText("Video loaded")

    def FunctionLoadImage(self):
        print(2)
        self.label_image.setText("Image loaded")

    def FunctionLoadFolder(self):
        print(3)
        self.label_folder.setText("Folder loaded")

    def FunctionBS(self):
        i = mean = std = 0
        frames = []

        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if ret != True:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)

            if i < 25:
                frames.append(gray)  
            else:
                if i == 25:
                    frames = np.array(frames)
                    mean = np.mean(frames, axis=0)
                    std = np.std(frames, axis=0)
                    std[std < 5] = 5
                diff = np.subtract(gray, mean)
                diff = np.absolute(diff)
                mask[diff > 5*std] = 255
                mask[diff <= 5*std] = 0

            result = cv2.bitwise_and(frame, frame, mask = mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            result = np.hstack((frame, mask, result))

            cv2.imshow("video", result)
            cv2.waitKey(int(1000/self.fps))
            i += 1

    def FunctionPreprocessing(self):
            print(1)

        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())