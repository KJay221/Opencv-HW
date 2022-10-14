from PyQt5 import QtWidgets
import UI
import sys, os, pprint
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
            cv2.drawChessboardCorners(obj[0], (11, 8), obj[1], True)
            img_height, img_width = obj[0].shape[:2]
            img_height = int(img_height/ 3)
            img_width = int(img_width/ 3)
            img = cv2.resize(obj[0], (img_height, img_width), interpolation=cv2.INTER_AREA)
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
        
    def camera(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        objpoints = []
        imgpoints = []
        
        folder = glob.glob(os.path.join(self.folder_path, "*"))
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

        # simple sort       
        def takeThird(elem):
            return elem[2]
        self.Img_list.sort(key = takeThird)
                
        # comboBox item
        for obj in self.Img_list:
            self.comboBox.addItem(obj[2])

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())