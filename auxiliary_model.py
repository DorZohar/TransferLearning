import keras
from gensim.corpora import WikiCorpus
from gensim.models.keyedvectors import KeyedVectors
from config import Config
import common
import numpy as np
import keras.backend as K
import pickle as pkl


class AuxiliaryModel:

    def __init__(self, conf, w2v=None, path=None):
        self.vectors = w2v
        self.vocab = pkl.load(open("LM_corpura/%s/%s" % (conf['lm_corpus'], conf['corpus__dict_file']), 'rb'))
        self.model = None
        self.conf = conf
        self.stacked_layer_names = []
        self.build_model()
        np.random.seed(self.conf['seed'])

        if path is None:
            self.train_gen = common.wiki_generator("LM_corpura/%s/%s" % (self.conf['lm_corpus'], self.conf['lm__train_file']),
                                                   len(self.vocab) + 1,
                                                   self.conf['lm__batch_size'],
                                                   self.conf['lm__max_sentence_len'])

            self.valid_gen = common.wiki_generator("LM_corpura/%s/%s" % (self.conf['lm_corpus'], self.conf['lm__valid_file']),
                                                   len(self.vocab) + 1,
                                                   self.conf['lm__batch_size'],
                                                   self.conf['lm__max_sentence_len'])

        else:
            self.load_from_file(path)

    def create_stacked_model(self, input_layer):

        embedding_matrix = common.create_embedding_matrix(self.vocab, self.vectors.vector_size, self.vectors)

        # Create an embedding layer
        embedding_layer = keras.layers.Embedding(input_dim=len(self.vocab) + 1,
                                                 output_dim=self.vectors.syn0.shape[1],
                                                 weights=[embedding_matrix],
                                                 name='EmbeddingLayer',
                                                 )(input_layer)
        self.stacked_layer_names.append('EmbeddingLayer')

        cur_fwd_layer = embedding_layer
        cur_bck_layer = embedding_layer

        # Create the stacked LSTMs
        for layer, layer_size in enumerate(self.conf['lm_lstm__hidden_sizes']):
            next_fwd_layer = keras.layers.LSTM(units=layer_size,
                                               activation=self.conf['lm__activation'],
                                               dropout=self.conf['lm_lstm__input_dropout'],
                                               recurrent_dropout=self.conf['lm_lstm__rec_dropout'],
                                               return_sequences=True,
                                               name="RecurrentFwd%d" % layer
                                               )
            next_bck_layer = keras.layers.LSTM(units=layer_size,
                                               activation=self.conf['lm__activation'],
                                               dropout=self.conf['lm_lstm__input_dropout'],
                                               recurrent_dropout=self.conf['lm_lstm__rec_dropout'],
                                               return_sequences=True,
                                               go_backwards=True,
                                               name="RecurrentBck%d" % layer
                                               )
            cur_fwd_layer = next_fwd_layer(cur_fwd_layer)
            cur_bck_layer = next_bck_layer(cur_bck_layer)

            self.stacked_layer_names.append("RecurrentFwd%d" % layer)
            self.stacked_layer_names.append("RecurrentBck%d" % layer)

        return cur_fwd_layer, cur_bck_layer

    def build_model(self):
        # Cannot build model twice
        assert self.model is None

        input_layer = keras.layers.Input(shape=(None, ),
                                         name='Input')

        cur_fwd_layer, cur_bck_layer = self.create_stacked_model(input_layer)

        cur_fwd_layer = keras.layers.Lambda(lambda x: x[:, :-1])(cur_fwd_layer)
        cur_bck_layer = keras.layers.Lambda(lambda x: x[:, 1:])(cur_bck_layer)

        input_zeroes = keras.layers.Lambda(lambda x: x[:, 0] * 0)(cur_fwd_layer)
        input_zeroes = keras.layers.RepeatVector(1)(input_zeroes)
        cur_fwd_layer = keras.layers.Concatenate(axis=1)([input_zeroes, cur_fwd_layer])
        cur_bck_layer = keras.layers.Concatenate(axis=1)([cur_bck_layer, input_zeroes])

        cur_layer = keras.layers.Concatenate()([cur_fwd_layer, cur_bck_layer])

        outputs = []
        losses = []
        metrics = []

        # Create the time-distributed dense layers (With dropout layers)
        for layer, layer_size in enumerate(self.conf['lm__dense_hidden_sizes']):

            next_layer = keras.layers.Dense(units=layer_size,
                                            activation=self.conf['lm__activation'],
                                            name="Dense%d" % layer
                                            )
            if layer == 0:
                next_layer = keras.layers.TimeDistributed(next_layer)
            cur_layer = next_layer(cur_layer)
            cur_layer = keras.layers.Dropout(self.conf['lm__dense_dropout'],
                                             name="Dropout%d" % layer
                                             )(cur_layer)

        # Final softmax layer for word classification
        output_layer = keras.layers.Dense(units=len(self.vocab) + 1,
                                          activation='softmax',
                                          name="OutputLayer")(cur_layer)

        outputs.append(output_layer)
        losses.append(common.sequential_sparse_categorical_crossentropy)
        metrics.append(common.perplexity)

        self.model = keras.models.Model(inputs=input_layer,
                                        outputs=outputs,
                                        )

        # Compile the model
        self.model.compile(loss=losses,
                           metrics=metrics,
                           optimizer=keras.optimizers.RMSprop(lr=self.conf['lm__learn_rate']),
                           sample_weight_mode="temporal",
                           )

        return

    def train(self):

        checkpoint = keras.callbacks.ModelCheckpoint(self.conf['lm__model_paths'],
                                                     save_best_only=True,
                                                     mode='min',
                                                     verbose=self.conf['verbose'])

        self.model.fit_generator(self.train_gen,
                                 steps_per_epoch=self.conf['lm__steps'] / self.conf['lm__batch_size'],
                                 epochs=self.conf['lm__epochs'],
                                 validation_data=self.valid_gen,
                                 validation_steps=self.conf['lm__steps'] / self.conf['lm__batch_size'] / 10,
                                 workers=4,
                                 callbacks=[checkpoint],
                                 verbose=1) #self.conf['verbose'])

    def test(self, gen_func):
        scores = self.model.evalute_generator(gen_func,
                                              steps_per_epoch=self.conf['lm__steps'],
                                              verbose=self.conf['verbose'])

        return scores

    def predict(self, gen_func):
        predictions = self.model.predict_generator(gen_func,
                                                   steps_per_epoch=self.conf['lm__steps'],
                                                   epochs=self.conf['lm__epochs'],
                                                   validation_steps=self.conf['lm__validation_steps'],
                                                   verbose=self.conf['verbose'])

        return predictions

    def load_from_file(self, path):
        keras.metrics.perplexity = common.perplexity
        self.model = keras.models.load_model(path)

    def get_hidden_layers(self, input_layer, trainable=False):

        cur_fwd_layer, cur_bck_layer = self.create_stacked_model(input_layer)
        output = keras.layers.Concatenate()([cur_fwd_layer, cur_bck_layer])

        new_model = keras.models.Model(inputs=input_layer, outputs=output)

        for layer_name in self.stacked_layer_names:
            new_model.get_layer(name=layer_name).set_weights(self.model.get_layer(layer_name).get_weights())
            new_model.get_layer(name=layer_name).trainable = trainable

        return new_model


if __name__ == '__main__':

    cfg = Config()
    w2v = KeyedVectors.load_word2vec_format(cfg['w2v_path'], binary=True)
    #wiki = WikiCorpus.load(cfg['wiki_path'])

    lm = AuxiliaryModel(cfg, w2v)
    lm.train()
