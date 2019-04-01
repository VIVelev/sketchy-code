import numpy as np
from keras import Model
from keras.layers import Input

__all__ = [
    'NSCInference',
]


class NSCInference:
    """Neural Sketch Coding - Inference
    
    Implements Inference for the Neural Sketch Coding (NSC) Model.
    
    Parameters:
    -----------
    neural_sketch_coding : the trained model
    word2idx : dictionary, mapping from vocabulary word to uniques indexes
    
    """

    def __init__(self, neural_sketch_coding, word2idx):
        self.neural_sketch_coding = neural_sketch_coding
        self.word2idx = word2idx
        self.idx2word = {val: key for key, val in word2idx.items()}

        self.inference_model = None
        self.build_inference_model()

    def build_inference_model(self):
        """Builds the Inference Model from the Neural Sketch Coding building blocks"""

        # Inputs
        num_hidden_neurons = self.neural_sketch_coding.num_hidden_neurons
        h_state_input_1 = Input((num_hidden_neurons[0],), name='h_state_input_1')
        c_state_input_1 = Input((num_hidden_neurons[0],), name='c_state_input_1')
        h_state_input_2 = Input((num_hidden_neurons[1],), name='h_state_input_2')
        c_state_input_2 = Input((num_hidden_neurons[1],), name='c_state_input_2')

        # Token Embeddings
        embedded_seq = self.neural_sketch_coding.sequence_decoder.embeddings(
            self.neural_sketch_coding.sequence_input
        )
        embedded_seq = self.neural_sketch_coding.sequence_decoder.embeddings_dropout(embedded_seq)

        # LSTM decoders
        output_tokens, h_state_1, c_state_1 = self.neural_sketch_coding.sequence_decoder.lstm_decoder_1(
            embedded_seq, initial_state=[h_state_input_1, c_state_input_1])
        output_tokens, h_state_2, c_state_2 = self.neural_sketch_coding.sequence_decoder.lstm_decoder_2(
            output_tokens, initial_state=[h_state_input_2, c_state_input_2])

        # Dense -> Softmax decoder
        output_tokens = self.neural_sketch_coding.sequence_decoder.dense_decoder(output_tokens)
        output_tokens = self.neural_sketch_coding.sequence_decoder.softmax_decoder(output_tokens)

        # Build The Model
        self.inference_model = Model(
            [self.neural_sketch_coding.sequence_input,
            h_state_input_1, c_state_input_1,
            h_state_input_2, c_state_input_2],

            [output_tokens,
            h_state_1, c_state_1,
            h_state_2, c_state_2]
        )

        return self

    def get_sketch_embedding(self, sketch):
        """Takes an image as an input and returns the fixed size feature vector"""
        return self.neural_sketch_coding.sketch_encoder.model.predict(np.expand_dims(sketch, 0))

    def get_initial_lstm_states(self, sketch):
        """Takes an image as an input and returns the context vectors (hidden and cell states of a LSTM network)"""

        states_model = Model(
            self.neural_sketch_coding.sequence_decoder.sketch_embedding_input,
            self.neural_sketch_coding.sequence_decoder.lstm_decoder_1(
                self.neural_sketch_coding.sequence_decoder.sketch_embedding_input)[1:]
        )

        sketch_emb = self.get_sketch_embedding(sketch)
        states = states_model.predict(sketch_emb)
        return states + [np.zeros((1, self.neural_sketch_coding.num_hidden_neurons[1]))]*2

    def greedy_search(self, sketch):
        """Greedy Search Inference"""

        # Get the context of the Sketch
        states_values = self.get_initial_lstm_states(sketch)

        # Start token
        target_seq = np.zeros((1, self.neural_sketch_coding.maxlen))
        target_seq[0, 0] = self.word2idx['<START>']

        # Init
        stop_condition = False
        decoded_tokens = []

        while not stop_condition:
            [output_tokens,
            h_state_1, c_state_1,
            h_state_2, c_state_2] = self.inference_model.predict(
                [target_seq] + states_values)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, 0, :])
            sampled_word = self.idx2word[sampled_token_index]

            if sampled_word == '{':
                decoded_tokens.append(' ')
                decoded_tokens.append(sampled_word)
                decoded_tokens.append('\n')

            elif sampled_word == '}':
                if self.idx2word[target_seq[0, 0]] != '}':
                    decoded_tokens.append('\n')
                decoded_tokens.append(sampled_word)
                decoded_tokens.append('\n')

            elif sampled_word == ',':
                decoded_tokens.append(sampled_word)
                decoded_tokens.append(' ')

            else:
                decoded_tokens.append(sampled_word)

            # Exit condition
            if sampled_word == '<END>' or len(decoded_tokens) > self.neural_sketch_coding.maxlen*3:
                stop_condition = True

            # Write sampled token
            target_seq = np.zeros((1, self.neural_sketch_coding.maxlen))
            target_seq[0, 0] = sampled_token_index

            states_values = [h_state_1, c_state_1, h_state_2, c_state_2]

        if '<END>' in decoded_tokens:
            decoded_tokens.remove('<END>')

        return ''.join(decoded_tokens)
