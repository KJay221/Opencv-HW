import sys, os
import UI
import glob
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets


class Window(QtWidgets.QWidget, UI.Ui_Form):
    
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button connect function
        self.ShowImages.clicked.connect(self.FunctionShowImages)

        # combo box
        self.Demo.addItem('Demo')
        self.Demo.addItem('Not Demo')

    def FunctionShowImages(self):
        if self.Demo.currentText() == 'Not Demo':
            self.training_dataset = tf.keras.utils.image_dataset_from_directory("./Dataset_CvDl_Hw2_Q5/training_dataset", labels = 'inferred', image_size=(224, 224), color_mode='rgb')
            self.validation_dataset = tf.keras.utils.image_dataset_from_directory("./Dataset_CvDl_Hw2_Q5/validation_dataset", labels = 'inferred', image_size=(224, 224), color_mode='rgb')
        else:
            self.inference_dataset = tf.keras.utils.image_dataset_from_directory("./inference_dataset", labels = 'inferred', image_size=(224, 224), color_mode='rgb')
            plt.figure(figsize=(7,7))
            class_names = self.inference_dataset.class_names
            dog = cat = 0
            count = 1
            for images, labels in self.inference_dataset.take(1):
                for i in range(len(images)):
                    if cat and dog:
                        break
                    if not dog and class_names[labels[i]] == "Dog":
                        plt.subplot(1, 2, count)
                        plt.imshow(images[i].numpy().astype("uint8"))
                        plt.title(class_names[labels[i]])
                        plt.axis("off")
                        dog = 1
                        count += 1
                    if not cat and class_names[labels[i]] == "Cat":
                        plt.subplot(1, 2, count)
                        plt.imshow(images[i].numpy().astype("uint8"))
                        plt.title(class_names[labels[i]])
                        plt.axis("off")
                        cat = 1
                        count += 1
                    
            # dog = cat = count = 1
            # # while not dog and not cat:
            # number = random.randint(0, len(image_list) - 1)
            # print(label_list[number])
            # plt.imshow(images[number].numpy().astype("uint8"))
            # plt.subplot(1, 10, count)
            # plt.axis("off")
            # count += 1
            plt.show()
        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())