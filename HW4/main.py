import sys, os
import UI
import cv2
import glob
import cv2.aruco as aruco
import numpy as np
from sklearn.decomposition import PCA
from PyQt5 import QtWidgets
from matplotlib import pyplot as plt


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
        self.PT.clicked.connect(self.FunctionPT)
        self.IR.clicked.connect(self.FunctionIR)
        self.CTRE.clicked.connect(self.FunctionCTRE)

    def FunctionLoadVideo(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.video = cv2.VideoCapture(path[0])
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.label_vedio.setText("Video loaded")

    def FunctionLoadImage(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.image_path = path[0]
        self.label_image.setText("Image loaded")

    def FunctionLoadFolder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory()
        self.folder = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
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

    def FunctionPT(self):
        img = cv2.imread(self.image_path)
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters_create()

        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if ret != True:
                break
            
            markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(frame, dictionary, parameters = parameters)
            if np.size(markerIds) != 4:
                continue

            idx = np.squeeze(np.where(markerIds == 1))
            pt1 = np.squeeze(markerCorners[idx[0]])[0]
            idx = np.squeeze(np.where(markerIds == 2))
            pt2 = np.squeeze(markerCorners[idx[0]])[1]
            idx = np.squeeze(np.where(markerIds == 3))
            pt3 = np.squeeze(markerCorners[idx[0]])[2]
            idx = np.squeeze(np.where(markerIds == 4))
            pt4 = np.squeeze(markerCorners[idx[0]])[3]

            pts_src = np.array([
                [0, 0],
                [img.shape[1], 0],
                [img.shape[1], 
                img.shape[0]],
                [0, img.shape[0]]
            ])
            
            pts_dst = np.array([
                [pt1[0], pt1[1]],
                [pt2[0], pt2[1]],
                [pt3[0], pt3[1]],
                [pt4[0], pt4[1]]
            ])

            h, status = cv2.findHomography(pts_src, pts_dst)
            temp = cv2.warpPerspective(img, h, (frame.shape[1], frame.shape[0]))
            cv2.fillConvexPoly(frame, pts_dst.astype(int), 0, 16)
            frame = frame + temp
            cv2.imshow("video", frame)
            cv2.waitKey(int(1000/self.fps))

    def Reconstruction(self, img):
        pca = PCA(n_components = 0.95)
        r, g, b = cv2.split(img)
        low_r = pca.fit_transform(r)
        re_r = pca.inverse_transform(low_r)
        low_g = pca.fit_transform(g)
        re_g = pca.inverse_transform(low_g)
        low_b = pca.fit_transform(b)
        re_b = pca.inverse_transform(low_b)
        clip_r = np.clip(re_r, a_min = 0, a_max = 255)
        clip_g = np.clip(re_g, a_min = 0, a_max = 255)
        clip_b = np.clip(re_b, a_min = 0, a_max = 255)
        new_img = (cv2.merge([clip_r, clip_g, clip_b])).astype(np.uint8)
        return new_img

    def FunctionIR(self):
        fig = plt.figure()
        
        # Image Reconstruction
        for id, image_path in enumerate(self.folder):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if id < 15:
                plt.subplot(4, 15, id + 1)
                plt.axis('off')
                plt.imshow(img)
                plt.subplot(4, 15, id + 16)
                plt.axis('off')
                plt.imshow(self.Reconstruction(img))
            else:
                plt.subplot(4, 15, id + 16)
                plt.axis('off')
                plt.imshow(img)
                plt.subplot(4, 15, id + 31)
                plt.axis('off')
                plt.imshow(self.Reconstruction(img))
        fig.text(0.01, 0.78, "Origin")
        fig.text(0.01, 0.58, "Reconstruction")
        fig.text(0.01, 0.38, "Origin")
        fig.text(0.01, 0.18, "Reconstruction")
        plt.show()

    def FunctionCTRE(self):
        error_list = []
        for image_path in self.folder:
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            new_img = self.Reconstruction(img)
            new_img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

            img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
            new_img_gray = cv2.normalize(new_img_gray, None, 0, 255, cv2.NORM_MINMAX)
            error = (np.sum((img_gray - new_img_gray)**2))**0.5
            error_list.append(error)
        print(error_list)

        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())