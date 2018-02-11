import keras
import common
import os
from config import Config
from constants import *
import numpy as np
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
import pickle as pkl

class Task:

    def __init__(self, conf):
        self.task_id = conf['cl_task_current']
        self.name = conf['cl_task__names'][self.task_id]
        self.num_classes = conf['cl_task__classes'][self.task_id]
        self.type = conf['cl_task__type'][self.task_id]
        self.conf = conf

    def get_corpus_path(self, mode):

        if mode == GENERATOR__TRAIN:
            filename = "train.txt"
        elif mode == GENERATOR__TEST:
            filename = "test.txt"
        else:
            filename = "validation.txt"
        path = "Tasks\\%s\\Corpus\\%s" % (self.name, filename)

        return path

    def get_model_path(self):
        base_path = "Tasks\\%s\\Models\\" % self.name
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        return "%s\\model_{epoch:02d}_{val_loss:.2f}.hdf5" % base_path

    # Converts any dataset format to a standard format:
    # List of tuples (X,y) for sequence tagging
    # List of tuples ([x_1, x_2.. x_n], y) for sentence classification
    # List of tuples ([x1_1, x1_2.. x1_n], [x2_1, x2_2.. x2_m], y) for two sentences classification
    def convert_format(self):
        sentences = []
        words = pkl.load(open("LM_corpura//%s//%s" % (cfg['lm_corpus'], cfg['corpus__dict_file']), 'rb'))
        word_dict = dict([(word, key) for key, word in enumerate(words, 1)])
        #word_dict = common.get_word_dict(self.conf['index2word_path'])

        if self.name == "pos_tagging":
            tags = set([tag for word, tag in treebank.tagged_words()])
            tag_index = {tag: idx for idx, tag in enumerate(tags, 1)}
            for sentence in treebank.tagged_sents():
                sent_words = [(word_dict[common.tokenize(w)], tag_index[t])
                              if common.tokenize(w) in word_dict else (0, tag_index[t])
                              for w, t in sentence]
                sentences.append(sent_words)

        return sentences

    def create_corpus(self, sentences):
        train, test = train_test_split(sentences,
                                       test_size=self.conf['corpus__test_fraction'],
                                       random_state=RANDOM_SEED)

        train, validation = train_test_split(train,
                                             test_size=self.conf['corpus__validation_fraction'] / (1 - self.conf['corpus__test_fraction']),
                                             random_state=RANDOM_SEED)

        for mode, split in [(GENERATOR__TRAIN, train), (GENERATOR__TEST, test), (GENERATOR__VALIDATION, validation)]:
            path = self.get_corpus_path(mode)
            with open(path, 'w') as fh:
                for sentence in split:
                    string = ["%d_%d" % (word, tag) for word, tag in sentence]
                    fh.write("%s\n" % " ".join(string))


    def line_to_words_tags(self, line, inputs, outputs):
        if self.type == TASK_TYPE__SEQUENCE_TAGGING:
            words = [int(word_tag.split('_')[0]) for word_tag in line]
            tags = [int(word_tag.split('_')[1]) for word_tag in line]
            inputs['Input'].append(words)
            outputs['OutputLayer'].append(tags)
        elif self.type == TASK_TYPE__SENTENCE_CLASSIFICATION:
            sentence, tag = line.split(CORPUS__SENTENCE_SEPERATOR)
            words = [int(word) for word in sentence]
            inputs['Input'].append(words)
            outputs['OutputLayer'].append(tag)
        else:
            sentence1, sentence2, tag = line.split(CORPUS__SENTENCE_SEPERATOR)
            words1 = [int(word) for word in sentence1]
            words2 = [int(word) for word in sentence2]
            inputs['Input1'].append(words1)
            inputs['Input2'].append(words2)
            outputs['OutputLayer'].append(tag)

        return inputs, outputs

    def process_batch(self, inputs, outputs, mode, max_len, is_auxiliary):

        ordered_input = {}

        for key, value in inputs.items():
            if value is None:
                continue
            ordered_input[key] = keras.preprocessing.sequence.pad_sequences(value,
                                                                            maxlen=max_len,
                                                                            padding='pre',
                                                                            truncating='post',
                                                                            )

        if is_auxiliary:
            assert self.type != TASK_TYPE__TWO_SENTENCES_CLASSIFICATION
            targets = np.expand_dims(ordered_input['Input'], axis=-1)
            weights = np.not_equal(ordered_input['Input'], 0).astype(np.float32)
            ordered_outputs = {'LMObjective': targets}
        else:
            if self.type == TASK_TYPE__SEQUENCE_TAGGING:
                weights = np.not_equal(ordered_input['Input'], 0).astype(np.float32)

                # TODO: Use to_categorical, then reshape (It flattens the array)
                padded_tags = keras.preprocessing.sequence.pad_sequences(outputs['OutputLayer'],
                                                                         maxlen=max_len,
                                                                         padding='pre',
                                                                         truncating='post',
                                                                         )

                padded_tags = keras.utils.to_categorical(padded_tags, self.num_classes)
                padded_tags = np.reshape(padded_tags, (len(outputs['OutputLayer']), max_len, self.num_classes))

                if mode != GENERATOR__TEST:
                    ordered_outputs = {'OutputLayer': padded_tags}
            else:
                weights = None
                if mode != GENERATOR__TEST:
                    tags = keras.utils.to_categorical(outputs['OutputLayer'], self.num_classes)
                    ordered_outputs = {'OutputLayer': tags}

        if mode != GENERATOR__TEST:
            return ordered_input, ordered_outputs, weights

        return ordered_input, weights

    @common.threadsafe_generator
    def generator(self, is_auxiliary=True, mode=GENERATOR__TRAIN, max_samples=None):

        file_path = self.get_corpus_path(mode)
        if is_auxiliary:
            max_len = self.conf['lm__max_sentence_len']
            batch_size = self.conf['lm__batch_size']
        else:
            max_len = self.conf['cl__max_sentence_len']
            batch_size = self.conf['cl__batch_size']

        inputs = {'Input': [], 'Input1': [], 'Input2': []}
        outputs = {'OutputLayer': []}
        i = 0
        while True:
            line_num = 0
            with open(file_path, 'r') as file:
                for line in file:
                    line_num += 1
                    if max_samples is not None and line_num > max_samples:
                        break
                    line = line.strip('\n').split(' ')
                    inputs, outputs = self.line_to_words_tags(line, inputs, outputs)
                    i += 1
                    if i >= batch_size:
                        yield self.process_batch(inputs, outputs, mode, max_len, is_auxiliary)
                        i = 0
                        inputs = {'Input': [], 'Input1': [], 'Input2': []}
                        outputs = {'OutputLayer': []}


if __name__ == "__main__":
    cfg = Config()

    task = Task(cfg)
    sents = task.convert_format()
    task.create_corpus(sents)
