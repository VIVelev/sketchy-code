from keras import Model
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D,
                          Reshape)
from keras.optimizers import RMSprop

from ..utils.config import IMAGE_SIZE

__all__ = [
    'SketchEncoder',
]


class SketchEncoder:
    """Sketch Encoder
    
    Sketch (Image) Enmbedding (Encoder) Model.
    
    Parameters:
    -----------
    embedding_dim : integer, the dimension in which to embed the sketch image and the tokens
    name : string, the name of the model, optional
    
    """

    def __init__(self, embedding_dim, name='sketch_encoder'):
        self.embedding_dim = embedding_dim
        self.name = name

        # Inputs
        self.image_input = Input(IMAGE_SIZE, name='image_input')

        # Conv 32
        self.conv_32_1 = Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv_32_1')
        self.conv_32_2 = Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv_32_2')
        self.maxpool_1 = MaxPool2D(pool_size=(2, 2), name='maxpool_1')
        self.conv_dropout_1 = Dropout(0.3, name='conv_dropout_1')

        # Conv 64
        self.conv_64_1 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv_64_1')
        self.conv_64_2 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv_64_2')
        self.maxpool_2 = MaxPool2D(pool_size=(2, 2), name='maxpool_2')
        self.conv_dropout_2 = Dropout(0.3, name='conv_dropout_2')

        # Conv 128
        self.conv_128_1 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv_128_1')
        self.conv_128_2 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv_128_2')
        self.maxpool_3 = MaxPool2D(pool_size=(2, 2), name='maxpool_3')
        self.conv_dropout_3 = Dropout(0.3, name='conv_dropout_3')

        # Flatten
        self.flatten = Flatten(name='flatten')

        # Dense -> ReLU 1
        self.dense_relu_1 = Dense(1024, activation='relu', name='dense_relu_1')
        self.dense_dropout_1 = Dropout(0.3, name='dense_dropout_1')

        # Dense -> ReLU 2
        self.dense_relu_2 = Dense(1024, activation='relu', name='dense_relu_2')
        self.dense_dropout_2 = Dropout(0.3, name='dense_dropout_2')

        # Dense -> ReLU encoder
        self.dense_relu_encoder = Dense(embedding_dim, activation='relu', name='dense_relu_encoder')
        self.embedding_reshapor = Reshape((1, embedding_dim), name='embedding_reshapor')

        self.model = None

    def build_model(self):
        """Builds a Keras Model to train/predict"""

        x = self.conv_32_1(self.image_input)
        x = self.conv_32_2(x)
        x = self.maxpool_1(x)
        x = self.conv_dropout_1(x)

        x = self.conv_64_1(x)
        x = self.conv_64_2(x)
        x = self.maxpool_2(x)
        x = self.conv_dropout_2(x)

        x = self.conv_128_1(x)
        x = self.conv_128_2(x)
        x = self.maxpool_3(x)
        x = self.conv_dropout_3(x)

        x = self.flatten(x)

        x = self.dense_relu_1(x)
        x = self.dense_dropout_1(x)

        x = self.dense_relu_2(x)
        x = self.dense_dropout_2(x)

        x = self.dense_relu_encoder(x)
        x = self.embedding_reshapor(x)

        self.model = Model(self.image_input, x, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
