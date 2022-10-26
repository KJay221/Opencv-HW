import UI
import sys
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PyQt5 import QtWidgets
from keras import optimizers
from keras.layers import Dense, Flatten, Dropout

class Window(QtWidgets.QWidget, UI.Ui_Form):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    demo = False
    

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.LoadImage1.clicked.connect(self.FunctionLoadImage1)
        self.LoadImage2.clicked.connect(self.FunctionLoadImage2)
        self.Keypoints.clicked.connect(self.FunctionKeypoints)
        self.MK.clicked.connect(self.FunctionMK)
        self.LoadImage.clicked.connect(self.FunctionLoadImage)
        self.STI.clicked.connect(self.FunctionSTI)
        self.SMS.clicked.connect(self.FunctionSMS)

        # load train data
        self.train_img = []
        with open('./cifar-10-python/cifar-10-batches-py/data_batch_1', 'rb') as file:
            train_dataset = pickle.load(file, encoding='bytes')
        for img_row in train_dataset[b'data']:
            img_R = img_row[0:1024].reshape((32, 32))
            img_G = img_row[1024:2048].reshape((32, 32))
            img_B = img_row[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))
            self.train_img.append(img)
        self.train_img = np.array(self.train_img)
        self.train_label = np.array(train_dataset[b'labels'])
        
        # load test data
        self.test_img = []
        with open('./cifar-10-python/cifar-10-batches-py/test_batch', 'rb') as file:
            test_dataset = pickle.load(file, encoding='bytes')
        self.test_labels = test_dataset[b'labels']
        for img_row in test_dataset[b'data']:
            img_R = img_row[0:1024].reshape((32, 32))
            img_G = img_row[1024:2048].reshape((32, 32))
            img_B = img_row[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))
            self.test_img.append(img)


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
        print(1)
        
    def FunctionSTI(self):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(self.classes[self.train_label[i]])
            plt.imshow(self.train_img[i])
        plt.show()

    def FunctionSMS(self):
        if self.demo == False:
            VGG19 = tf.keras.applications.VGG19(
                include_top = False, input_shape = (32, 32, 3), classes = 10)
            model = tf.keras.models.Sequential()
            model.add(VGG19)
            model.add(Flatten())
            model.add(Dense(1024,activation = 'relu'))
            model.add(Dropout(.25))
            model.add(Dense(1024,activation = 'relu'))
            model.add(Dropout(.25))
            model.add(Dense(256,activation = 'relu'))
            model.add(Dense(10,activation = 'softmax'))
            optimizer = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
            label = tf.cast(self.train_label, dtype=tf.int32)
            label = tf.squeeze(label)
            label = tf.one_hot(label, depth=10)
            model.compile(optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
            train_history = model.fit(
                x = self.train_img, y = label, epochs = 30, batch_size = 32, verbose = 1)
            self.model.save('./model/model.h5')
        else:
            print(1)
        print(model.summary())

    def f(self):
        print(1)        
        
        

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())