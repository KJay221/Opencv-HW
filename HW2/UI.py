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
        Form.resize(279, 467)
        self.groupBox1 = QtWidgets.QGroupBox(Form)
        self.groupBox1.setEnabled(True)
        self.groupBox1.setGeometry(QtCore.QRect(30, 30, 221, 411))
        self.groupBox1.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox1.setFlat(False)
        self.groupBox1.setCheckable(False)
        self.groupBox1.setObjectName("groupBox1")
        self.LoadImage1 = QtWidgets.QPushButton(self.groupBox1)
        self.LoadImage1.setGeometry(QtCore.QRect(10, 60, 201, 41))
        self.LoadImage1.setAutoDefault(False)
        self.LoadImage1.setDefault(False)
        self.LoadImage1.setObjectName("LoadImage1")
        self.LoadImage2 = QtWidgets.QPushButton(self.groupBox1)
        self.LoadImage2.setGeometry(QtCore.QRect(10, 150, 201, 41))
        self.LoadImage2.setAutoDefault(False)
        self.LoadImage2.setDefault(False)
        self.LoadImage2.setObjectName("LoadImage2")
        self.Keypoints = QtWidgets.QPushButton(self.groupBox1)
        self.Keypoints.setGeometry(QtCore.QRect(10, 240, 201, 41))
        self.Keypoints.setAutoDefault(False)
        self.Keypoints.setDefault(False)
        self.Keypoints.setObjectName("Keypoints")
        self.MK = QtWidgets.QPushButton(self.groupBox1)
        self.MK.setGeometry(QtCore.QRect(10, 330, 201, 41))
        self.MK.setAutoDefault(False)
        self.MK.setDefault(False)
        self.MK.setObjectName("MK")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "2022 CvDI Hw1_4"))
        self.groupBox1.setTitle(_translate("Form", "4. SIFT"))
        self.LoadImage1.setText(_translate("Form", "LoadImage1"))
        self.LoadImage2.setText(_translate("Form", "LoadImage2"))
        self.Keypoints.setText(_translate("Form", "4.1 Keypoints"))
        self.MK.setText(_translate("Form", "4.2 Matched Keypoints"))
