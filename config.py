import pickle
from constants import *

defualt_items = {
# File location parameters
    'w2v_path': "C:\\Wiki\\wiki.word2vec.bin",
    'wiki_path': 'C:\\Wiki\\wiki.corpus',
    'tfidf_path': 'C:\\Wiki\\wiki.tfidf.model',
    'lm_corpus': 'Brown',
    'corpus__data_file': 'lm_input.txt',
    'corpus__dict_file': 'lm_dict.txt',
    'lm__train_file': 'lm_input_train.txt',
    'lm__test_file': 'lm_input_test.txt',
    'lm__valid_file': 'lm_input_validation.txt',
    'lm__model_paths': 'LM_Models/model_{epoch:02d}_{val_loss:.2f}.hdf5',

#################################
# Language Model                #
#################################

# Language Model Parameters
    'lm__dense_dropout': 0.2,
    'lm__dense_hidden_sizes': [],
    'lm__activation': 'tanh',
    'lm__epochs': 50,
    'lm__steps': 29730,
    'lm__validation_steps': 10,
    'lm__batch_size': 100,
    'lm__max_sentence_len': 40,
    'lm__learn_rate': 0.001,

# Languague Model's LSTM parameters
    'lm_lstm__hidden_sizes': [150],
    'lm_lstm__input_dropout': 0.2,
    'lm_lstm__rec_dropout': 0.1,

#################################
# Classifier                    #
#################################

    'cl__steps': 1000,
    'cl__validation_steps': 100,
    'cl__test_steps': 100,
    'cl__batch_size': 200,
    'cl__epochs': 10,
    'cl__max_sentence_len': 40,


# Classifier's tasks
    'cl_task__names': ['pos_tagging',
                       'chunking',
                       'ner',
                       'sentiment_analysis'],
    'cl_task__classes': [],
    'cl_task__type': [TASK_TYPE__SEQUENCE_TAGGING,
                      TASK_TYPE__SEQUENCE_TAGGING,
                      TASK_TYPE__SEQUENCE_TAGGING,
                      TASK_TYPE__SENTENCE_CLASSIFICATION,

                      ],
    'cl_task_current': 0,

# Classifier structure

    'cl_struct__trainable': False,
    'cl_struct__attention': False,
    'cl_struct__bidirectional': True,
    'cl_struct__lm_class': 'Brown',
    'cl_struct__lm_file': 'model_10_4.04.hdf5',
    'cl_struct__lstm_sizes': [40],
    'cl_struct__dense_sizes': [150, 150],
    'cl_struct__rec_dropout': 0.1,
    'cl_struct__input_dropout': 0.2,
    'cl_struct__dense_dropout': 0.3,
    'cl_struct__activation': 'tanh',
    'cl_struct__learn_rate': 0.001,

# General Parameters
    'verbose': 1,
    'seed': 42,


#################################
# Corpus                        #
#################################
    'corpus__min_word_freq': 3,
    'corpus__train_fraction': 0.7,
    'corpus__test_fraction': 0.15,

}


class Config:
    def __init__(self, load_path=None):
        self.items = defualt_items
        if load_path is not None:
            self.load(load_path)

    def __getitem__(self, item):
        return self.items[item]

    def save(self, file_path):
        pickle.dump(self.items, file_path)

    def load(self, file_path):
        self.items = pickle.load(file_path)

