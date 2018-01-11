import keras
from gensim.models import KeyedVectors
import common
from config import Config
from auxiliary_model import AuxiliaryModel
import pickle as pkl
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk.corpus import treebank
from constants import *
from task import Task

class Classifier:

    def __init__(self, cfg):
        lm_path = "LM_Models/%s/%s" % (cfg['cl_struct__lm_class'], cfg['cl_struct__lm_file'])
        w2v = KeyedVectors.load_word2vec_format(cfg['w2v_path'], binary=True)
        self.lm = AuxiliaryModel(cfg, w2v, path=lm_path)
        self.words = pkl.load(open("LM_corpura//%s//%s" % (cfg['lm_corpus'], cfg['corpus__dict_file']), 'rb'))
        self.word_dict = dict([(word, key) for key, word in enumerate(self.words, 1)])
        self.model = None
        self.conf = cfg
        self.task = Task(cfg)

    def build_model(self):

        input_layer = keras.layers.Input(shape=(None,),
                                         name='Input')

        auxiliary_output = self.lm.get_hidden_layers(input_layer, trainable=self.conf['cl_struct__trainable'])

        cur_layer = auxiliary_output
        return_sequences = True

        for layer, layer_size in enumerate(self.conf['cl_struct__lstm_sizes']):
            if layer == len(self.conf['cl_struct__lstm_sizes']) - 1 and not self.conf['cl_struct__attention']:
                return_sequences = False
            next_layer = keras.layers.LSTM(layer_size,
                                           activation=self.conf['cl_struct__activation'],
                                           recurrent_dropout=self.conf['cl_struct__rec_dropout'],
                                           dropout=self.conf['cl_struct__input_dropout'],
                                           return_sequences=return_sequences,
                                           name="Rec%d" % layer)

            if self.conf['cl_struct__bidirectional']:
                cur_layer = keras.layers.Bidirectional(next_layer)(cur_layer)
            else:
                cur_layer = next_layer(cur_layer)

        # Create the time-distributed dense layers (With dropout layers)
        for layer, layer_size in enumerate(self.conf['cl_struct__dense_sizes']):

            next_layer = keras.layers.Dense(units=layer_size,
                                            activation=self.conf['cl_struct__activation'],
                                            name="Dense%d" % layer
                                            )
            if layer == 0 and return_sequences:
                next_layer = keras.layers.TimeDistributed(next_layer)
            cur_layer = next_layer(cur_layer)
            cur_layer = keras.layers.Dropout(self.conf['cl_struct__dense_dropout'],
                                             name="Dropout%d" % layer
                                             )(cur_layer)

        # Final softmax layer for word classification
        output_layer = keras.layers.Dense(units=self.task,
                                          activation='softmax',
                                          name="OutputLayer")(cur_layer)

        self.model = keras.models.Model(input_layer, output_layer)

        if self.task.type == TASK_TYPE__SEQUENCE_TAGGING:
            self.model.compile(loss="crossentropy_loss",
                               metrics=[common.sequence_accuracy],
                               optimizer=keras.optimizers.RMSprop(lr=self.conf['cl_struct__learn_rate']),
                               sample_weight_mode="temporal",
                               )
        else:
            self.model.compile(loss="crossentropy_loss",
                               metrics=["accuracy"],
                               optimizer=keras.optimizers.RMSprop(lr=self.conf['cl_struct__learn_rate']),
                               )

    def train(self, train_gen, valid_gen):
        checkpoint = keras.callbacks.ModelCheckpoint(self.task.get_model_path(),
                                                     save_best_only=True,
                                                     mode='min',
                                                     verbose=self.conf['verbose'])

        self.model.fit_generator(train_gen,
                                 steps_per_epoch=self.conf['cl__steps'] / self.conf['cl__batch_size'],
                                 epochs=self.conf['cl__epochs'],
                                 validation_data=valid_gen,
                                 validation_steps=self.conf['cl__validation_steps'] / self.conf['lm__batch_size'],
                                 workers=4,
                                 callbacks=[checkpoint],
                                 verbose=1)

    def test(self, test_gen):
        scores = self.model.evalute_generator(test_gen,
                                              steps_per_epoch=self.conf['cl__test_steps'],
                                              verbose=self.conf['verbose'])

        return scores

    def read_data(self, tagged_sentences):
        features = []
        tags = []

        for sentence in tagged_sentences:
            sent_words = [self.word_dict[common.tokenize(w)]
                          if common.tokenize(w) in self.word_dict else len(self.words)
                          for w, t in sentence]
            sent_tags = [t for w, t in sentence]
            sent_representations = self.lm.predict(np.asarray(sent_words))
            sent_representations = np.squeeze(sent_representations, axis=1)
            features.append(sent_representations)
            tags.append(sent_tags)

        all_tags = set(np.concatenate(tags))

        tags_enum = [(tag, idx) for idx, tag in enumerate(all_tags)]
        tag_dict = dict(tags_enum)

        tags = [[tag_dict[tag] for tag in sent_tags] for sent_tags in tags]

        #features = np.concatenate(features)
        tags = np.asarray(tags)
        tags = keras.utils.to_categorical(tags)

        X_train, X_test, y_train, y_test = train_test_split(features, tags, test_size=0.2, random_state=42)

        for size in [0.01, 0.05, 0.1, 0.2]:
            print("Using %d examples" % (int(X_train.shape[0] * size)))
            svm = SVC(kernel='linear')
            svm.fit(X_train[:int(X_train.shape[0] * size)], y_train[:int(X_train.shape[0] * size)])
            score = svm.score(X_test, y_test)
            print("for %.2f%% of the data: %.2f%% accuracy" % (100*size, 100*score))


if __name__ == '__main__':
    cfg = Config()
    c = Classifier(cfg)
    c.read_data(treebank.tagged_sents())


