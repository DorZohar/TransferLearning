import pickle

defualt_items = {
# File location parameters
    'w2v_path': "C:\\Wiki\\wiki.word2vec.bin",
    'wiki_path': 'C:\\Wiki\\wiki.corpus',
    'tfidf_path': 'C:\\Wiki\\wiki.tfidf.model',
    'corpus__data_file': 'C:\\Wiki\\lm_input.txt',
    'corpus__dict_file': 'C:\\Wiki\\lm_dict.txt',
    'lm__train_file': 'C:\\Wiki\\lm_input_train.txt',
    'lm__test_file': 'C:\\Wiki\\lm_input_test.txt',
    'lm__valid_file': 'C:\\Wiki\\lm_input_validation.txt',
    'lm__model_paths': 'LM_Models\\model_{epoch:02d}_{val_loss:.2f}.hdf5',

#################################
# Language Model                #
#################################

# Language Model Parameters
    'lm__dense_dropout': 0.2,
    'lm__dense_hidden_sizes': [],
    'lm__activation': 'tanh',
    'lm__epochs': 50,
    'lm__steps': 29730, #95,
    'lm__classes': 5000,
    'lm__validation_steps': 10,
    'lm__batch_size': 500,
    'lm__max_sentence_len': 40,

# Languague Model's LSTM parameters
    'lm_lstm__hidden_sizes': [150],
    'lm_lstm__input_dropout': 0.2,
    'lm_lstm__rec_dropout': 0.1,

#################################
# Classifier                    #
#################################

# Classifier's tasks
    'cl_task__names': ['pos_tagging', ],
    'cl_task__classes': [],
    'cl_task__': [],
    'cl_task_current': 0,

# Classifier structure

    


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

