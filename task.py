import keras
from constants import *
import numpy as np

class Task:

    def __init__(self, conf):
        self.task_id = conf['cl_task_current']
        self.name = conf['cl_task__names'][self.task_id]
        self.num_classes = conf['cl_task__classes'][self.task_id]
        self.type = conf['cl_task__type'][self.task_id]
        self.conf = conf

    def get_corpus_path(self):
        pass

    # Converts any dataset format to a standard format:
    # List of tuples (X,y) for sequence tagging
    # List of tuples ([x_1, x_2.. x_n], y) for sentence classification
    # List of tuples ([x1_1, x1_2.. x1_n], [x2_1, x2_2.. x2_m], y) for two sentences classification
    def convert_format(self):
        pass

    def create_corpus(self):
        pass

    def get_model_path(self):
        pass

    def generator(self, is_auxiliary=True, mode=GENERATOR__TRAIN):

        file_path = self.get_corpus_path()
        if is_auxiliary:
            max_len = self.conf['lm__max_sentence_len']
            batch_size = self.conf['lm__batch_size']
        else:
            max_len = self.conf['cl__max_sentence_len']
            batch_size = self.conf['cl__batch_size']

        batch_sentences = []
        i = 0
        while True:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip('\n').split(' ')
                    line = [int(word_id) for word_id in line]
                    pos = 0
                    batch_sentences.append(line)
                    i += 1
                    pos += max_len
                    if i >= batch_size:

                        sequences = keras.preprocessing.sequence.pad_sequences(batch_sentences,
                                                                               maxlen=max_len,
                                                                               padding='pre',
                                                                               truncating='post',
                                                                               )

                        targets = np.expand_dims(sequences, axis=-1)
                        weights = np.not_equal(sequences, 0).astype(np.float32)

                        if mode == GENERATOR__TEST:
                            yield {'Input': sequences}
                        else:
                            yield {'Input': sequences}, targets, weights
                        i = 0
                        batch_sentences = []
