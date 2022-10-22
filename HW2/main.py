from PyQt5 import QtWidgets
import UI
import sys
import cv2

class Window(QtWidgets.QWidget, UI.Ui_Form):
    # declare path
    Image1_path = ''
    Image2 = ''

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.LoadImage1.clicked.connect(self.FunctionLoadImage1)
        self.LoadImage2.clicked.connect(self.FunctionLoadImage2)
        self.Keypoints.clicked.connect(self.FunctionKeypoints)
        self.MK.clicked.connect(self.FunctionMK)

    def FunctionLoadImage1(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.Image1_path = path[0]
    
    def FunctionLoadImage2(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.Image2_path = path[0]

    def FunctionKeypoints(self):
        img1 = cv2.imread(self.Image1_path)
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        showimg = cv2.drawKeypoints(gray, kp, 0, (255, 0, 0),
                                 flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(self.Image1_path.split("/")[-1], showimg)

    def FunctionMK(self):
        img1 = cv2.imread(self.Image1_path)
        img2 = cv2.imread(self.Image2_path)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
        
        showimg = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("img", showimg)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())