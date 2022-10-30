import random
import UI
import sys, os
import cv2
import keras
import PIL
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
from torchvision import transforms


class Window(QtWidgets.QWidget, UI.Ui_Form):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    learning_rate = 0.01
    epochs = 30

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button connect function
        self.LoadImage.clicked.connect(self.FunctionLoadImage)
        self.STI.clicked.connect(self.FunctionSTI)
        self.SMS.clicked.connect(self.FunctionSMS)
        self.SDA.clicked.connect(self.FunctionSDA)
        self.SAAL.clicked.connect(self.FunctionSAAL)
        self.Inference.clicked.connect(self.FunctionInference)

        # combo box
        self.Demo.addItem('Demo')
        self.Demo.addItem('Not Demo')

        # load data
        (self.trainX, self.trainY), (self.testX, self.testY) = cifar10.load_data()
        self.x_train = self.trainX.astype('float32')/255
        self.x_test = self.testX.astype('float32')/255
        self.y_train = np_utils.to_categorical(self.trainY)
        self.y_test = np_utils.to_categorical(self.testY)

        # get model
        if self.Demo.currentText() == 'Not Demo':
            VGG19 = tf.keras.applications.VGG19(
                include_top = False, input_shape = (32, 32, 3), classes = 10)
            self.model = tf.keras.models.Sequential()
            self.model.add(VGG19)
            self.model.add(Flatten())
            self.model.add(Dense(1024, activation = 'relu'))
            self.model.add(Dropout(.25))
            self.model.add(Dense(1024, activation = 'relu'))
            self.model.add(Dropout(.25))
            self.model.add(Dense(256, activation = 'relu'))
            self.model.add(Dense(10, activation = 'softmax'))

            optimizer = optimizers.SGD(learning_rate = self.learning_rate, decay = 1e-6, momentum = 0.9, nesterov = True)
            self.model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
            learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', 
                patience = 3, verbose = 1, factor = 0.6, min_lr = 0.00001)
            result = self.model.fit(
                x = self.x_train, y = self.y_train, epochs = self.epochs, batch_size = 32, validation_split = 0.15, verbose = 1, callbacks = [learning_rate_reduction])
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

    def FunctionLoadImage(self):
        self.test_number = random.randint(0, 9999)
        img_show = cv2.cvtColor(self.testX[self.test_number], cv2.COLOR_BGR2RGB)
        img_path = './pic/' + str(self.test_number) + '.png'
        cv2.imwrite(img_path, img_show)
        pix = QPixmap(img_path)
        pix = pix.scaled(315,315)
        item = QtWidgets.QGraphicsPixmapItem(pix)
        scene = QtWidgets.QGraphicsScene(self)
        scene.setSceneRect(0, 0, 315, 315)  
        scene.addItem(item)
        self.Image.setScene(scene)
        os.remove(img_path)
        self.Label.setText('Label = ' + self.classes[self.testY[self.test_number][0]])
        
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
        print(self.model.summary())

    def FunctionSDA(self):
        img_path = './pic/' + str(self.test_number) + '.png'
        cv2.imwrite(img_path, self.testX[self.test_number])
        img_pil = PIL.Image.open(img_path, mode = 'r')
        img_pil = img_pil.convert('RGBA')

        # RandomRotation
        transform = transforms.Compose([transforms.RandomRotation(degrees = 90)])
        img_RandomRotation = transform(img_pil)
        img_RandomRotation = np.array(img_RandomRotation)
        img_RandomRotation = cv2.resize(img_RandomRotation, (320, 320))

        # RandomResizedCrop
        transform = transforms.Compose([transforms.RandomResizedCrop(size = 224, scale = (0.5, 0.5))])
        img_RandomResizedCrop = transform(img_pil)
        img_RandomResizedCrop = np.array(img_RandomResizedCrop)
        img_RandomResizedCrop = cv2.resize(img_RandomResizedCrop, (320, 320))

        # RandomHorizontalFlip
        transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1)])
        img_RandomHorizontalFlip = transform(img_pil)
        img_RandomHorizontalFlip = np.array(img_RandomHorizontalFlip)
        img_RandomHorizontalFlip = cv2.resize(img_RandomHorizontalFlip, (320, 320))

        img_show = np.concatenate((img_RandomRotation, img_RandomResizedCrop, img_RandomHorizontalFlip), axis = 1)
        cv2.imshow('img', img_show)
        os.remove(img_path)

    def FunctionSAAL(self):
        img_show = np.concatenate((cv2.imread('./pic/accuracy.png'), cv2.imread('./pic/loss.png')), axis = 1)
        cv2.imshow('img', img_show)

    def FunctionInference(self):
        img = np.array([self.x_test[self.test_number]])
        tag = []
        for i in range(10):
            tag.append(self.classes[i])
        prediction = self.model.predict(img)
        self.PredictionLabel.setText('Prediction Label = ' + self.classes[prediction.argmax()])
        self.Confidence.setText('Confidence = ' + "{:.4f}".format(prediction[0][prediction.argmax()]))


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())