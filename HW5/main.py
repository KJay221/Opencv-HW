import sys
import UI
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from train import Resnet50


class Window(QtWidgets.QWidget, UI.Ui_Form):
    model = Resnet50().model()
    
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # button connect function
        self.ShowImages.clicked.connect(self.FunctionShowImages)
        self.SD.clicked.connect(self.FunctionSD)
        self.SMS.clicked.connect(self.FunctionSMS)

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
            for images, labels in self.inference_dataset:
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
            plt.show()
    
    def FunctionSMS(self):
        print(self.model.summary())
    
    def FunctionSD(self):
        if self.Demo.currentText() == 'Not Demo':
            self.training_dataset = tf.keras.utils.image_dataset_from_directory("./Dataset_CvDl_Hw2_Q5/training_dataset", labels = 'inferred', image_size=(224, 224), color_mode='rgb')
            
            class_names = self.training_dataset.class_names
            dog = cat = 0
            for images, labels in self.training_dataset:
                for i in range(len(images)):
                    if class_names[labels[i]] == "Dog":
                        dog += 1
                    else:
                        cat += 1
            
            y = [cat, dog]
            plt.figure()
            plot = plt.bar(["Cat", "Dog"], y)
            plt.yticks(np.arange(0, 15000, 1000))
            for value in plot:
                height = value.get_height()
                plt.text(value.get_x() + value.get_width()/2., 1.002*height,'%d' % int(height), ha='center', va='bottom')
            plt.ylabel("Number of images")
            plt.xlabel("Class")
            plt.title('Class Distribution')
            plt.savefig('./img/Class_Distribution.png')
            plt.show()
        else:
            img = mpimg.imread('./img/Class_Distribution.png')
            plt.imshow(img)
            plt.axis("off")
            plt.show()

        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = Window()
    ui.show()
    sys.exit(app.exec_())