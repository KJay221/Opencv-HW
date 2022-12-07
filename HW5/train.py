import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

IMAGE_SIZE = [224, 224]

class Resnet50(object):
    def __init__(self):
        super(Resnet50, self).__init__()
        

    def model(self):
        resnet = tf.keras.applications.resnet50.ResNet50(input_shape=IMAGE_SIZE + [3], include_top=False)
        for layer in resnet.layers:
            layer.trainable = False
        x = Flatten()(resnet.output)
        prediction = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=resnet.input, outputs=prediction)
                
        return model