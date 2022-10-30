import random
import UI
import sys, os
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PyQt5.QtGui import QPixmap
from PyQt5 import QtWidgets
from keras import optimizers
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau


class Window(QtWidgets.QWidget, UI.Ui_Form):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    demo = False
    learning_rate = 0.01
    epochs = 30

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button connect function
        self.LoadImage1.clicked.connect(self.FunctionLoadImage1)
        self.LoadImage2.clicked.connect(self.FunctionLoadImage2)
        self.Keypoints.clicked.connect(self.FunctionKeypoints)
        self.MK.clicked.connect(self.FunctionMK)
        self.LoadImage.clicked.connect(self.FunctionLoadImage)
        self.STI.clicked.connect(self.FunctionSTI)
        self.SMS.clicked.connect(self.FunctionSMS)
        self.Inference.clicked.connect(self.FunctionInference)

        # load data
        (self.trainX, self.trainY), (self.testX, self.testY) = cifar10.load_data()
        self.x_train = self.trainX.astype('float32')/255
        self.x_test = self.testX.astype('float32')/255
        self.y_train = np_utils.to_categorical(self.trainY)
        self.y_test = np_utils.to_categorical(self.testY)

        # reduce size
        # self.x_train = self.x_train[0:1000]
        # self.y_train = self.y_train[0:1000]


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

    def FunctionLoadImage(self):
        number = random.randint(0, 9999)
        img_path = './pic/' + str(number) + '.png'
        cv2.imwrite(img_path, self.testX[number])
        pix = QPixmap(img_path)
        pix = pix.scaled(315,315)
        item = QtWidgets.QGraphicsPixmapItem(pix)
        scene = QtWidgets.QGraphicsScene(self)
        scene.setSceneRect(0, 0, 315, 315)  
        scene.addItem(item)
        self.Image.setScene(scene)
        os.remove(img_path)
        self.Label.setText('Label = ' + self.classes[self.testY[number][0]])
        
    def FunctionSTI(self):
        number = random.randint(0, 49991)
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(self.classes[self.trainY[i + number][0]])
            plt.imshow(self.trainX[i + number])
        plt.show()

    def FunctionSMS(self):
        if self.demo == False:
            VGG19 = tf.keras.applications.VGG19(
                include_top = False, input_shape = (32, 32, 3), classes = 10)
            self.model = tf.keras.models.Sequential()
            self.model.add(VGG19)
            self.model.add(Flatten())
            self.model.add(Dense(1024,activation = 'relu'))
            self.model.add(Dropout(.25))
            self.model.add(Dense(1024,activation = 'relu'))
            self.model.add(Dropout(.25))
            self.model.add(Dense(256,activation = 'relu'))
            self.model.add(Dense(10,activation = 'softmax'))

            optimizer = optimizers.SGD(learning_rate = self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                patience=3, verbose=1, factor=0.6, min_lr=0.00001)
            result = self.model.fit(
                x = self.x_train, y = self.y_train, epochs = self.epochs, batch_size = 32, validation_split=0.15, verbose = 1, callbacks = [learning_rate_reduction])
            self.model.save('./model/model.h5')
            plt.plot(result.history['accuracy'])
            plt.plot(result.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left') 
            plt.savefig('./pic/accuracy.png')   
            plt.close()

            plt.plot(result.history['loss']) 
            plt.plot(result.history['val_loss']) 
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left') 
            plt.savefig('./pic/loss.png')
            plt.close()
        else:
            self.model = keras.models.load_model('./model/model.h5')
        print(self.model.summary())

    def FunctionInference(self):
        img = np.array([self.x_test[5]])
        tag = []
        for i in range(10):
            tag.append(self.classes[i])
        tag[0] = 'plane'
        tag[1] = 'car'
        pred = self.model.predict(img)
        #print(pred[0])

        fig = plt.figure()
        fig.set_size_inches(12, 6)
        plt.subplot(1, 2, 1)
        plt.imshow(img[0])
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.bar(tag, pred[0])
        plt.show()       

        

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())