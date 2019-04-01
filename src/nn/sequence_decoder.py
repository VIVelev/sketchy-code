from keras import Model
from keras.layers import (LSTM, Activation, Dense, Dropout, Embedding, Input,
                          TimeDistributed)
from keras.optimizers import RMSprop


__all__ = [
    'SequenceDecoder',
]


class SequenceDecoder:
    def __init__(self, maxlen, embedding_dim, voc_size, num_hidden_neurons, name='sequence_decoder'):
        self.sequence_input = Input((maxlen,), name='sequence_input')
        self.sketch_embedding_input = Input((1, embedding_dim), name='sketch_embedding_input')
        
        self.embeddings = Embedding(
            voc_size,
            embedding_dim,
            input_length=maxlen,
            mask_zero=True,
            name='embeddings'
        )

        self.embeddings_dropout = Dropout(0.3, name='embeddings_dropout')

        self.lstm_decoder_1 = LSTM(num_hidden_neurons[0], return_sequences=True, return_state=True, name='lstm_decoder_1')
        self.lstm_decoder_2 = LSTM(num_hidden_neurons[1], return_sequences=True, return_state=True, name='lstm_decoder_2')

        self.dense_decoder = TimeDistributed(Dense(voc_size, name='dense_layer'), name='dense_decoder')
        self.softmax_decoder = TimeDistributed(Activation('softmax', name='softmax_layer'), name='softmax_decoder')

        self.name = name
        self.model = None

    def build_model(self):
        x = self.embeddings(self.sequence_input)
        x = self.embeddings_dropout(x)

        _, h_state, c_state = self.lstm_decoder_1(self.sketch_embedding_input)

        x, _, _ = self.lstm_decoder_1(x, initial_state=[h_state, c_state])
        x, _, _ = self.lstm_decoder_2(x)

        x = self.dense_decoder(x)
        x = self.softmax_decoder(x)

        self.model = Model([self.sequence_input, self.sketch_embedding_input], x, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
