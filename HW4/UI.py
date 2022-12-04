# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(368, 766)
        self.groupBox_1 = QtWidgets.QGroupBox(Form)
        self.groupBox_1.setEnabled(True)
        self.groupBox_1.setGeometry(QtCore.QRect(30, 250, 311, 81))
        self.groupBox_1.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_1.setFlat(False)
        self.groupBox_1.setCheckable(False)
        self.groupBox_1.setObjectName("groupBox_1")
        self.BS = QtWidgets.QPushButton(self.groupBox_1)
        self.BS.setGeometry(QtCore.QRect(10, 30, 291, 41))
        self.BS.setAutoDefault(False)
        self.BS.setDefault(False)
        self.BS.setObjectName("BS")
        self.LoadVideo = QtWidgets.QPushButton(Form)
        self.LoadVideo.setGeometry(QtCore.QRect(30, 10, 311, 41))
        self.LoadVideo.setAutoDefault(False)
        self.LoadVideo.setDefault(False)
        self.LoadVideo.setObjectName("LoadVideo")
        self.LoadImage = QtWidgets.QPushButton(Form)
        self.LoadImage.setGeometry(QtCore.QRect(30, 90, 311, 41))
        self.LoadImage.setAutoDefault(False)
        self.LoadImage.setDefault(False)
        self.LoadImage.setObjectName("LoadImage")
        self.LoadFolder = QtWidgets.QPushButton(Form)
        self.LoadFolder.setGeometry(QtCore.QRect(30, 170, 311, 41))
        self.LoadFolder.setAutoDefault(False)
        self.LoadFolder.setDefault(False)
        self.LoadFolder.setObjectName("LoadFolder")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 360, 311, 131))
        self.groupBox_2.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setCheckable(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.Preprocessing = QtWidgets.QPushButton(self.groupBox_2)
        self.Preprocessing.setGeometry(QtCore.QRect(10, 30, 291, 41))
        self.Preprocessing.setAutoDefault(False)
        self.Preprocessing.setDefault(False)
        self.Preprocessing.setObjectName("Preprocessing")
        self.VideoTracking = QtWidgets.QPushButton(self.groupBox_2)
        self.VideoTracking.setGeometry(QtCore.QRect(10, 80, 291, 41))
        self.VideoTracking.setAutoDefault(False)
        self.VideoTracking.setDefault(False)
        self.VideoTracking.setObjectName("VideoTracking")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setEnabled(True)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 520, 311, 81))
        self.groupBox_3.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_3.setFlat(False)
        self.groupBox_3.setCheckable(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.PT = QtWidgets.QPushButton(self.groupBox_3)
        self.PT.setGeometry(QtCore.QRect(10, 30, 291, 41))
        self.PT.setAutoDefault(False)
        self.PT.setDefault(False)
        self.PT.setObjectName("PT")
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setEnabled(True)
        self.groupBox_4.setGeometry(QtCore.QRect(30, 620, 311, 131))
        self.groupBox_4.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_4.setFlat(False)
        self.groupBox_4.setCheckable(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.IR = QtWidgets.QPushButton(self.groupBox_4)
        self.IR.setGeometry(QtCore.QRect(10, 30, 291, 41))
        self.IR.setAutoDefault(False)
        self.IR.setDefault(False)
        self.IR.setObjectName("IR")
        self.CTRE = QtWidgets.QPushButton(self.groupBox_4)
        self.CTRE.setGeometry(QtCore.QRect(10, 80, 291, 41))
        self.CTRE.setAutoDefault(False)
        self.CTRE.setDefault(False)
        self.CTRE.setObjectName("CTRE")
        self.label_vedio = QtWidgets.QLabel(Form)
        self.label_vedio.setGeometry(QtCore.QRect(30, 60, 311, 17))
        self.label_vedio.setObjectName("label_vedio")
        self.label_image = QtWidgets.QLabel(Form)
        self.label_image.setGeometry(QtCore.QRect(30, 140, 311, 17))
        self.label_image.setObjectName("label_image")
        self.label_folder = QtWidgets.QLabel(Form)
        self.label_folder.setGeometry(QtCore.QRect(30, 220, 311, 17))
        self.label_folder.setObjectName("label_folder")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "CVDL 2022 HW2"))
        self.groupBox_1.setTitle(_translate("Form", "1. Background Subtraction"))
        self.BS.setText(_translate("Form", "1.1 Background Subtraction"))
        self.LoadVideo.setText(_translate("Form", "Load Video"))
        self.LoadImage.setText(_translate("Form", "Load Image"))
        self.LoadFolder.setText(_translate("Form", "Load Folder"))
        self.groupBox_2.setTitle(_translate("Form", "2. Optical Flow"))
        self.Preprocessing.setText(_translate("Form", "2.1 Preprocessing"))
        self.VideoTracking.setText(_translate("Form", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("Form", "3. Perspective Transform"))
        self.PT.setText(_translate("Form", "3.1 Perspective Transform"))
        self.groupBox_4.setTitle(_translate("Form", "5. PCA"))
        self.IR.setText(_translate("Form", "4.1 Image Reconstruction"))
        self.CTRE.setText(_translate("Form", "4.2 Compute the reconstruction error"))
        self.label_vedio.setText(_translate("Form", "No video loaded"))
        self.label_image.setText(_translate("Form", "No image loaded"))
        self.label_folder.setText(_translate("Form", "No folder loaded"))
