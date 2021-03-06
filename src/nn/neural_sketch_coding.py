from keras import Model
from keras.layers import Input
from keras.optimizers import RMSprop

from ..utils.config import IMAGE_SIZE
from .sequence_decoder import SequenceDecoder
from .sketch_encoder import SketchEncoder

__all__ = [
    'NeuralSketchCoding',
]


class NeuralSketchCoding:
    """Neural Sketch Coding
    
    The full model for generating code from sketch.
    
    Parameters:
    -----------
    embedding_dim : integer, the dimension in which to embed the sketch image and the tokens
    maxlen : integer, the maximum code length
    voc_size : integer, number of unique tokens in the vocabulary
    num_hidden_neurons : list with length of 2, specifying the number of hidden neurons in the LSTM decoders
    name : string, the name of the model, optional
    
    """

    def __init__(self, embedding_dim, maxlen, voc_size, num_hidden_neurons, name='neural_sketch_coding'):
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.voc_size = voc_size
        self.num_hidden_neurons = num_hidden_neurons
        self.name = name

        # Inputs
        self.image_input = Input(IMAGE_SIZE, name='image_input')
        self.sequence_input = Input((maxlen,), name='sequence_input')

        # Encoder / Decoder
        self.sketch_encoder = SketchEncoder(embedding_dim).build_model()
        self.sequence_decoder = SequenceDecoder(maxlen, embedding_dim, voc_size, num_hidden_neurons).build_model()

        self.model = None

    def build_model(self):
        """Builds a Keras Model to train/predict"""

        sketch_embedding = self.sketch_encoder.model(self.image_input)
        sequence_output = self.sequence_decoder.model([self.sequence_input, sketch_embedding])

        self.model = Model([self.image_input, self.sequence_input], sequence_output, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
