from PyQt5 import QtWidgets
import UI
import sys, os
import cv2
import glob
import numpy as np

class Window(QtWidgets.QWidget, UI.Ui_Form):
    # declare path
    folder_path = ''
    ImageL_path = ''
    ImageR_path = ''
    Img_list = []
 
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button click connect
        self.LoadFolder.clicked.connect(self.browsedirectory)
        self.LoadImageL.clicked.connect(self.browsefileL)
        self.LoadImageR.clicked.connect(self.browsefileR)
        self.FindCorners.clicked.connect(self.FunctionFindCorners)
        self.FindIntrinsic.clicked.connect(self.FunctionFindIntrinsic)
        self.FindExtrinsic.clicked.connect(self.FunctionFindExtrinsic)
        self.FindDistortion.clicked.connect(self.FunctionFindDistortion)
        self.ShowResult.clicked.connect(self.FunctionShowResult)
        self.SWOB.clicked.connect(lambda : self.FunctionSW("onbord"))
        self.SWV.clicked.connect(lambda : self.FunctionSW("vertical"))
        self.SDM.clicked.connect(self.FunctionSDM)

    # Load Image
    def browsedirectory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory()
        self.folder_path = path
        self.camera()

    def browsefileL(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.ImageL_path = path[0]
            
    def browsefileR(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.ImageR_path = path[0]        

    # 1. function
    def FunctionFindCorners(self):
        for obj in self.Img_list:
            img = obj[0].copy()
            cv2.drawChessboardCorners(img, (11, 8), obj[1], True)
            img_height, img_width = img.shape[:2]
            img_height = int(img_height/ 3)
            img_width = int(img_width/ 3)
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
            cv2.imshow(obj[2] ,img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def FunctionFindIntrinsic(self):
        print("Intrinsic:")
        print(self.mtx)

    def FunctionFindExtrinsic(self):
        index = 0
        for i in range(len(self.Img_list)):
            if self.Img_list[i][2] == self.comboBox.currentText():
                index = i
        
        rvec2 = cv2.Rodrigues(self.rvecs[index])
        print("Extrinsic:")
        print(np.append(rvec2[0], self.tvecs[index], axis = 1))

    def FunctionFindDistortion(self):
        print("Distortion:")
        print(self.dist)

    def FunctionShowResult(self):
        for obj in self.Img_list:
            img = obj[0].copy()
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

            img_height, img_width = dst.shape[:2]
            img_height = int(img_height/ 3)
            img_width = int(img_width/ 3)
            dst = cv2.resize(dst, (img_height, img_width), interpolation=cv2.INTER_AREA)
            dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)

            img_height, img_width = img.shape[:2]
            img_height = int(img_height/ 3)
            img_width = int(img_width/ 3)
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            show = np.concatenate([dst, img], axis = 1)
            cv2.imshow(obj[2], show)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def camera(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        objpoints = []
        imgpoints = []
        
        self.Img_list = []
        folder = []
        folder = glob.glob(os.path.join(self.folder_path, "*.bmp")) + glob.glob(os.path.join(self.folder_path, "*.png"))
        for obj in folder:
            filename = obj.split("/")[-1]
            img = cv2.imread(obj)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners)
                self.Img_list.append([img, corners2, filename])

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        for i in range(len(self.Img_list)):
            self.Img_list[i].append(self.rvecs[i])
            self.Img_list[i].append(self.tvecs[i])

        # simple sort       
        def takeThird(elem):
            return elem[2]
        self.Img_list.sort(key = takeThird)
                
        # comboBox item
        self.comboBox.clear()
        for obj in self.Img_list:
            self.comboBox.addItem(obj[2])

        
    # 2. function
    def FunctionSW(self, type):
        if type == "onbord":
            libpath = self.folder_path + '/Q2_lib/alphabet_lib_onboard.txt'
            show = cv2.FileStorage(libpath, cv2.FILE_STORAGE_READ)
        else:
            libpath = self.folder_path + '/Q2_lib/alphabet_lib_vertical.txt'
            show = cv2.FileStorage(libpath, cv2.FILE_STORAGE_READ)

        for obj in self.Img_list:
            img = obj[0].copy()
            rvec = obj[3].copy()
            tvec = obj[4].copy()
        
            def GenWord(m, n):
                Word = show.getNode(m).mat()
                xpost = (n % 3) * 3
                ypost = int(n / 3) * 3
                for i in Word:
                    src = np.array(i, np.float64)
                    src[0][0] += 7 - xpost
                    src[0][1] += 5 - ypost
                    src[1][0] += 7 - xpost
                    src[1][1] += 5 - ypost
                    cameraMatrix = self.mtx
                    result = cv2.projectPoints(src, rvec, tvec, cameraMatrix, None)

                    result = tuple(map(tuple, result[0]))
                    start = tuple(map(int, result[0][0]))
                    end = tuple(map(int, result[1][0]))
                    cv2.line(img, start, end, (255, 0, 0), 5)
            
            text = str(self.lineEdit.text())
            text = text.upper()
            for i ,word in enumerate(text):
                if(word.isalpha()):
                    GenWord(word, i)

            img_height, img_width = img.shape[:2]
            img_height = int(img_height/ 3)
            img_width = int(img_width/ 3)
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)

            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imshow(obj[2], img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    # 3. function
    def FunctionSDM(self):
        imgL = cv2.imread(self.ImageL_path)
        imgR = cv2.imread(self.ImageR_path)

        def resize(img):
            img_height, img_width = img.shape[:2]
            img_height = img_height - img_height % 256
            img_width = img_width - img_width % 256
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
            return img
        imgL = resize(imgL)
        imgR = resize(imgR)

        imgL_gray = cv2.cvtColor(imgL.copy(), cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR.copy(), cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        img = stereo.compute(imgL_gray, imgR_gray)
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_height, img_width = img.shape[:2]
        img_height = int(img_height/ 4)
        img_width = int(img_width/ 4)
        img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
        cv2.imshow("img", img)

        # imgR
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        img_height, img_width = imgR.shape[:2]
        img_height = int(img_height/ 4)
        img_width = int(img_width/ 4)
        imgRshow = cv2.resize(imgR, (img_height, img_width), interpolation=cv2.INTER_AREA)
        cv2.imshow("imgR", imgRshow)
        
        # set mouse callback
        def mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                nonlocal imgRshow
                imgRshow_circle = imgRshow.copy()
                imgRshow_circle = cv2.circle(imgRshow_circle, (x, y), 3, (255, 0, 0), 10)
                cv2.imshow("imgR", imgRshow_circle)
        
        # imgL
        cv2.namedWindow('imgL')
        cv2.setMouseCallback('imgL', mouse_click)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        img_height, img_width = imgL.shape[:2]
        img_height = int(img_height/ 4)
        img_width = int(img_width/ 4)
        imgLshow = cv2.resize(imgL, (img_height, img_width), interpolation=cv2.INTER_AREA)
        cv2.imshow("imgL", imgLshow)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())