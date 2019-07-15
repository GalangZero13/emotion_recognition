from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation 
from keras.layers.core import Flatten 
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K 

class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # #1: first CONV > RELU > CONV > RELU > POOL
        model.add(Conv2D(32, (3, 3), padding = "same", kernel_initializer = "he_normal", input_shape = inputShape))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3, 3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        
        # #2: second CONV > RELU > CONV > RELU > POOL
        model.add(Conv2D(64, (3, 3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3, 3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # #3: third CONV > RELU > CONV > RELU > POOL
        model.add(Conv2D(128, (3, 3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(128, (3, 3), kernel_initializer = "he_normal", padding = "same"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # #4: first set of FC > RELU layers
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # #5: second set of FC > RELU layers
        model.add(Dense(64, kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # #6: softmax classifier
        model.add(Dense(classes, kernel_initializer = "he_normal"))
        model.add(Activation("softmax"))

        return model