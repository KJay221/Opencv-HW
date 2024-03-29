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
        Form.resize(619, 467)
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 20, 221, 411))
        self.groupBox_2.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setCheckable(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.LoadImage = QtWidgets.QPushButton(self.groupBox_2)
        self.LoadImage.setGeometry(QtCore.QRect(10, 50, 201, 41))
        self.LoadImage.setAutoDefault(False)
        self.LoadImage.setDefault(False)
        self.LoadImage.setObjectName("LoadImage")
        self.STI = QtWidgets.QPushButton(self.groupBox_2)
        self.STI.setGeometry(QtCore.QRect(10, 110, 201, 41))
        self.STI.setAutoDefault(False)
        self.STI.setDefault(False)
        self.STI.setObjectName("STI")
        self.SDA = QtWidgets.QPushButton(self.groupBox_2)
        self.SDA.setGeometry(QtCore.QRect(10, 230, 201, 41))
        self.SDA.setAutoDefault(False)
        self.SDA.setDefault(False)
        self.SDA.setObjectName("SDA")
        self.SAAL = QtWidgets.QPushButton(self.groupBox_2)
        self.SAAL.setGeometry(QtCore.QRect(10, 290, 201, 41))
        self.SAAL.setAutoDefault(False)
        self.SAAL.setDefault(False)
        self.SAAL.setObjectName("SAAL")
        self.SMS = QtWidgets.QPushButton(self.groupBox_2)
        self.SMS.setGeometry(QtCore.QRect(10, 170, 201, 41))
        self.SMS.setAutoDefault(False)
        self.SMS.setDefault(False)
        self.SMS.setObjectName("SMS")
        self.Inference = QtWidgets.QPushButton(self.groupBox_2)
        self.Inference.setGeometry(QtCore.QRect(10, 350, 201, 41))
        self.Inference.setAutoDefault(False)
        self.Inference.setDefault(False)
        self.Inference.setObjectName("Inference")
        self.Image = QtWidgets.QGraphicsView(Form)
        self.Image.setGeometry(QtCore.QRect(270, 110, 320, 320))
        self.Image.setObjectName("Image")
        self.Confidence = QtWidgets.QLabel(Form)
        self.Confidence.setGeometry(QtCore.QRect(280, 10, 211, 30))
        self.Confidence.setObjectName("Confidence")
        self.PredictionLabel = QtWidgets.QLabel(Form)
        self.PredictionLabel.setGeometry(QtCore.QRect(280, 40, 331, 30))
        self.PredictionLabel.setObjectName("PredictionLabel")
        self.Label = QtWidgets.QLabel(Form)
        self.Label.setGeometry(QtCore.QRect(280, 70, 331, 30))
        self.Label.setObjectName("Label")
        self.Demo = QtWidgets.QComboBox(Form)
        self.Demo.setGeometry(QtCore.QRect(510, 10, 101, 25))
        self.Demo.setObjectName("Demo")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "2022 CvDI Hw1_5"))
        self.groupBox_2.setTitle(_translate("Form", "5. IMG DL part"))
        self.LoadImage.setText(_translate("Form", "Load Image"))
        self.STI.setText(_translate("Form", "1. Show Train Images"))
        self.SDA.setText(_translate("Form", "3. Show Data Augmentation"))
        self.SAAL.setText(_translate("Form", "4. Show Accuracy and Loss"))
        self.SMS.setText(_translate("Form", "2. Show Model Structure"))
        self.Inference.setText(_translate("Form", "5. Inference"))
        self.Confidence.setText(_translate("Form", "Confidence = "))
        self.PredictionLabel.setText(_translate("Form", "Prediction Label = "))
        self.Label.setText(_translate("Form", "Label = "))
