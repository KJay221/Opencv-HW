from PyQt5 import QtWidgets
import UI
import sys, os
import cv2
import glob

class Window(QtWidgets.QWidget, UI.Ui_Form):
    # declare path
    folder_path = ''
    ImageL_path = ''
    ImageR_path = ''

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button click connect
        self.LoadFolder.clicked.connect(self.browsedirectory)
        self.LoadImageL.clicked.connect(self.browsefileL)
        self.LoadImageR.clicked.connect(self.browsefileR)
        
    # Load Image
    def browsedirectory(self):
        path = QtWidgets.QFileDialog.getExistingDirectory()
        self.folder_path = path
        self.FunctionFindCorners()

    def browsefileL(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.ImageL_path = path[0]
            
    def browsefileR(self):
        path = QtWidgets.QFileDialog.getOpenFileName()
        self.ImageR_path = path[0]        

    def FunctionFindCorners(self):
        folder = glob.glob(os.path.join(self.folder_path, "*"))
        for index, obj in enumerate(folder):
            img = cv2.imread(obj)
            ret, corners = cv2.findChessboardCorners(img, (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
                img_height, img_width = img.shape[:2]
                img_height = int(img_height/ 3)
                img_width = int(img_width/ 3)
                img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
                cv2.imshow(str(index+1) ,img)


        

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())