import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

IMAGE_SIZE = [224, 224]
LEARNING_RATE = 8e-5
EPOCHS = 1

class Resnet50(object):
    def __init__(self):
        super(Resnet50, self).__init__()
        
    def model(self):
        resnet = tf.keras.applications.resnet50.ResNet50(input_shape=IMAGE_SIZE + [3], include_top=False)
        # # take advantage of the pre-trained layers
        # for layer in resnet.layers:
        #     layer.trainable = False
        x = Flatten()(resnet.output)
        prediction = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=resnet.input, outputs=prediction)       
        return model
    
    def train(self, model, training_dataset):
        # Focal Loss
        loss_function = tfa.losses.SigmoidFocalCrossEntropy(alpha = 0.4, gamma = 1.0)
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer = optimizer, loss = loss_function, metrics = ['accuracy'])
        model.fit(training_dataset, epochs = EPOCHS, batch_size = 32, verbose = 1)
        model.save('./model/model_focal.h5')

        # Binary Cross Entropy
        loss_function = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer = optimizer, loss = loss_function, metrics = ['accuracy'])
        model.fit(training_dataset, epochs = EPOCHS, batch_size = 32, verbose = 1)
        model.save('./model/model_binary.h5')