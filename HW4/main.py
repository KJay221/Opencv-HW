import sys
import UI
import cv2
import numpy as np
from PyQt5 import QtWidgets


class Window(QtWidgets.QWidget, UI.Ui_Form):
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 35
    params.maxArea = 90
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.maxCircularity = 0.9
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button connect function
        self.LoadVideo.clicked.connect(self.FunctionLoadVideo)
        self.LoadImage.clicked.connect(self.FunctionLoadImage)
        self.LoadFolder.clicked.connect(self.FunctionLoadFolder)
        self.BS.clicked.connect(self.FunctionBS)
        self.Preprocessing.clicked.connect(self.FunctionPreprocessing)
        self.VideoTracking.clicked.connect(self.FunctionVideoTracking)

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
        if self.video.isOpened():
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detector = cv2.SimpleBlobDetector_create(self.params)
            keypoints = detector.detect(gray)
            print(keypoints[1].pt)

            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                frame = cv2.rectangle(frame, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 1)
                frame = cv2.line(frame, (x, y - 6), (x, y + 6), (0, 0, 255), 1)
                frame = cv2.line(frame, (x - 6, y), (x + 6, y), (0, 0, 255), 1)
            cv2.imshow("Image", frame)

    def FunctionVideoTracking(self):
        lk_params = dict(
                winSize = (15,15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        ret, frame = self.video.read()
        gray_origin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.SimpleBlobDetector_create(self.params)
        keypoints = detector.detect(gray_origin)

        p0 = np.array([[[kp.pt[0], kp.pt[1]]] for kp in keypoints]).astype(np.float32)
        mask = np.zeros_like(frame)

        while(self.video.isOpened()):

            ret, frame = self.video.read()
            if ret != True:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            p1, st, err = cv2.calcOpticalFlowPyrLK(gray_origin, gray, p0, None, **lk_params)

            good_new = p1[st==1]
            good_old = p0[st==1]

            for (new,old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 3)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            result = cv2.add(frame, mask)
            
            cv2.imshow("video", result)
            cv2.waitKey(int(1000/self.fps))
            gray_origin = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())