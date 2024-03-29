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
        Form.resize(989, 558)
        self.groupBox1 = QtWidgets.QGroupBox(Form)
        self.groupBox1.setEnabled(True)
        self.groupBox1.setGeometry(QtCore.QRect(20, 70, 221, 411))
        self.groupBox1.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox1.setFlat(False)
        self.groupBox1.setCheckable(False)
        self.groupBox1.setObjectName("groupBox1")
        self.LoadFolder = QtWidgets.QPushButton(self.groupBox1)
        self.LoadFolder.setGeometry(QtCore.QRect(20, 90, 181, 41))
        self.LoadFolder.setAutoDefault(False)
        self.LoadFolder.setDefault(False)
        self.LoadFolder.setObjectName("LoadFolder")
        self.LoadImageL = QtWidgets.QPushButton(self.groupBox1)
        self.LoadImageL.setGeometry(QtCore.QRect(20, 190, 181, 41))
        self.LoadImageL.setAutoDefault(False)
        self.LoadImageL.setDefault(False)
        self.LoadImageL.setObjectName("LoadImageL")
        self.LoadImageR = QtWidgets.QPushButton(self.groupBox1)
        self.LoadImageR.setGeometry(QtCore.QRect(20, 300, 181, 41))
        self.LoadImageR.setAutoDefault(False)
        self.LoadImageR.setDefault(False)
        self.LoadImageR.setObjectName("LoadImageR")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setGeometry(QtCore.QRect(270, 70, 221, 411))
        self.groupBox_2.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setCheckable(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.FindCorners = QtWidgets.QPushButton(self.groupBox_2)
        self.FindCorners.setGeometry(QtCore.QRect(20, 40, 181, 41))
        self.FindCorners.setAutoDefault(False)
        self.FindCorners.setDefault(False)
        self.FindCorners.setObjectName("FindCorners")
        self.FindIntrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.FindIntrinsic.setGeometry(QtCore.QRect(20, 100, 181, 41))
        self.FindIntrinsic.setAutoDefault(False)
        self.FindIntrinsic.setDefault(False)
        self.FindIntrinsic.setObjectName("FindIntrinsic")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 150, 201, 121))
        self.groupBox_5.setObjectName("groupBox_5")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_5)
        self.comboBox.setGeometry(QtCore.QRect(40, 30, 121, 31))
        self.comboBox.setCurrentText("")
        self.comboBox.setObjectName("comboBox")
        self.FindExtrinsic = QtWidgets.QPushButton(self.groupBox_5)
        self.FindExtrinsic.setGeometry(QtCore.QRect(10, 70, 181, 41))
        self.FindExtrinsic.setAutoDefault(False)
        self.FindExtrinsic.setDefault(False)
        self.FindExtrinsic.setObjectName("FindExtrinsic")
        self.FindDistortion = QtWidgets.QPushButton(self.groupBox_2)
        self.FindDistortion.setGeometry(QtCore.QRect(20, 290, 181, 41))
        self.FindDistortion.setAutoDefault(False)
        self.FindDistortion.setDefault(False)
        self.FindDistortion.setObjectName("FindDistortion")
        self.ShowResult = QtWidgets.QPushButton(self.groupBox_2)
        self.ShowResult.setGeometry(QtCore.QRect(20, 350, 181, 41))
        self.ShowResult.setAutoDefault(False)
        self.ShowResult.setDefault(False)
        self.ShowResult.setObjectName("ShowResult")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setEnabled(True)
        self.groupBox_3.setGeometry(QtCore.QRect(510, 70, 221, 411))
        self.groupBox_3.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_3.setFlat(False)
        self.groupBox_3.setCheckable(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.SWOB = QtWidgets.QPushButton(self.groupBox_3)
        self.SWOB.setGeometry(QtCore.QRect(20, 190, 181, 41))
        self.SWOB.setAutoDefault(False)
        self.SWOB.setDefault(False)
        self.SWOB.setObjectName("SWOB")
        self.SWV = QtWidgets.QPushButton(self.groupBox_3)
        self.SWV.setGeometry(QtCore.QRect(20, 280, 181, 41))
        self.SWV.setAutoDefault(False)
        self.SWV.setDefault(False)
        self.SWV.setObjectName("SWV")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit.setGeometry(QtCore.QRect(20, 90, 181, 41))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setEnabled(True)
        self.groupBox_4.setGeometry(QtCore.QRect(750, 70, 221, 411))
        self.groupBox_4.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_4.setFlat(False)
        self.groupBox_4.setCheckable(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.SDM = QtWidgets.QPushButton(self.groupBox_4)
        self.SDM.setGeometry(QtCore.QRect(20, 190, 181, 41))
        self.SDM.setAutoDefault(False)
        self.SDM.setDefault(False)
        self.SDM.setObjectName("SDM")

        self.retranslateUi(Form)
        self.comboBox.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "2022 CvDI Hw1"))
        self.groupBox1.setTitle(_translate("Form", "Load Image"))
        self.LoadFolder.setText(_translate("Form", "Load Folder"))
        self.LoadImageL.setText(_translate("Form", "Load Image_L"))
        self.LoadImageR.setText(_translate("Form", "Load Image_R"))
        self.groupBox_2.setTitle(_translate("Form", "1. Calibration"))
        self.FindCorners.setText(_translate("Form", "1.1 Find Corners"))
        self.FindIntrinsic.setText(_translate("Form", "1.2 Find Intrinsic"))
        self.groupBox_5.setTitle(_translate("Form", "1.3 Find Extrinsic"))
        self.FindExtrinsic.setText(_translate("Form", "1.3 Find Extrinsic"))
        self.FindDistortion.setText(_translate("Form", "1.4 Find Distortion "))
        self.ShowResult.setText(_translate("Form", "Show Result"))
        self.groupBox_3.setTitle(_translate("Form", "2. Augmented Reality"))
        self.SWOB.setText(_translate("Form", "2.1 Show Words on Board"))
        self.SWV.setText(_translate("Form", "2.2 Show Words Vertically"))
        self.groupBox_4.setTitle(_translate("Form", "3. Stereo Disparity Map"))
        self.SDM.setText(_translate("Form", "3.1 Stereo Disparity Map"))
