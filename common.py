import numpy as np
from config import Config
import threading
import keras
import keras.backend as K
import pickle as pkl
import os
import re
import tensorflow as tf
import time
from constants import *

def sequential_sparse_categorical_crossentropy(y_true, y_pred):
    #
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    #
    # non_zeros = K.cast(K.not_equal(y_true, K.constant(0)), tf.bool)
    # indices = tf.where(non_zeros)
    #
    # y_true = K.expand_dims(K.expand_dims(K.gather(y_true, indices), -1), 0)
    # y_pred = K.expand_dims(K.expand_dims(K.gather(y_pred, indices), -1), 0)

    return K.sparse_categorical_crossentropy(y_true, y_pred)


def perplexity(y_true, y_pred):
    return K.pow(K.constant(2.0), K.mean(sequential_sparse_categorical_crossentropy(y_true, y_pred)))


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


# class SampledSoftmax(keras.layers.Layer):
#
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(SampledSoftmax, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.Ws = self.add_weight(name='softmax_weights',
#                                   shape=(input_shape[1], self.output_dim),
#                                   initializer='uniform',
#                                   trainable=True)
#
#         self.biases = self.add_weight(name='sofmax_biases',
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#
#         super(SampledSoftmax, self).build(input_shape)
#
#     def call(self, x, **kwargs):
#         train_softmax = tf.nn.sampled_softmax_loss(self.Ws, self.biases, )
#
#
#         return K.in_train_phase()
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)


def sequence_accuracy(y_true, y_pred):
    non_zeros = (K.not_equal(y_true, K.constant(0)))
    true_positions = K.any(non_zeros, axis=1)
    return K.mean(K.equal(y_true[true_positions], y_pred[true_positions]))


def get_file_with_suffixes(file, suffixes):
    base, ext = os.path.splitext(file)
    return ["%s_%s%s" % (base, suffix, ext) for suffix in suffixes]


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def wiki_generator(file_path, max_idx, batch_size, max_len, is_test=False):

    batch_sentences = []
    i = 0
    while True:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip('\n').split(' ')
                line = [int(word_id) for word_id in line]
                pos = 0
                while pos < len(line):
                    sentence = line[pos:pos+max_len]
                    batch_sentences.append(sentence)
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
                        #weights = weights / np.expand_dims(np.sum(weights, axis=1), 1)

                        if is_test:
                            yield {'Input': sequences}
                        else:
                            yield {'Input': sequences}, targets, weights
                        i = 0
                        batch_sentences = []


def preprocessWiki(wiki, word_dictionary, max_idx, output):

    with open(output, 'w') as file:
        for text in wiki.get_texts():
            sentence = [str(word_dictionary[word].index) if word in word_dictionary else str(max_idx)
                        for word in text]

            file.write('%s\n' % ' '.join(sentence))


def create_embedding_matrix(index_to_word, vector_size, init_vectors=None):
    embedding_matrix = np.zeros((len(index_to_word) + 1, vector_size))

    if init_vectors is not None:
        for key, word in enumerate(index_to_word):
            if word in init_vectors:
                embedding_matrix[key] = init_vectors[word]

    return embedding_matrix


def tokenize(word):

    word = word.lower()
    if re.match("^-?[0-9]+(.[0-9]+)?$", word):
        word = ":NUM:"

    return word


def preprocess_brown(min_freq, min_sentence_len, output, index_to_word_output):
    from nltk.corpus import brown
    from collections import Counter

    print("Total number of words: %d" % len(brown.words()))

    counter = Counter([tokenize(word) for word in brown.words()])
    word_frequencies = counter.most_common()

    print("Number of unique tokenized words: %d" % len(word_frequencies))

    frequencies = [freq for word, freq in word_frequencies if freq >= min_freq]
    word_by_frequencies = [word for word, freq in word_frequencies if freq >= min_freq]
    word_dict = {key: word for word, key in enumerate(word_by_frequencies, 1)}
    vocab_size = len(word_by_frequencies)
    print("Number of unique tokenized words with at least %d appearances: %d" % (min_freq, vocab_size))
    print("Total number of words after removals: %d, percentage: %.2f%%" % (sum(frequencies),
                                                                            100 * sum(frequencies) / len(brown.words())))

    with open(output, 'w') as file:
        for text in brown.sents():
            if len(text) < min_sentence_len:
                continue
            sentence = [str(word_dict[tokenize(word)]) if tokenize(word) in word_dict else str(vocab_size)
                        for word in text]

            file.write('%s\n' % ' '.join(sentence))

    pkl.dump(word_by_frequencies, open(index_to_word_output, 'wb'))


def split_dataset(file, train_fraction, test_fraction, randomize=False):
    num_lines = sum(1 for line in open(file, 'r'))
    train_lines = int(num_lines * train_fraction)
    test_lines = int(num_lines * test_fraction)

    file_names = get_file_with_suffixes(file, ['train', 'test', 'validation'])
    files = [open(file_name, 'w') for file_name in file_names]

    for i, line in enumerate(open(file, 'r')):
        if randomize:
            idx = np.random.choice([0, 1, 2], p=[train_fraction, test_fraction, 1 - train_fraction - test_fraction])
        else:
            if i < train_lines:
                idx = 0
            elif i < train_lines + test_lines:
                idx = 1
            else:
                idx = 2

        files[idx].write(line)

    for f in files:
        f.close()

    for file in file_names:
        line_lengths = [len(line.split(' ')) for line in open(file, 'r')]
        print("%s Statistics:" % file)
        print("\tNumber of lines: %d" % len(line_lengths))
        print("\tAverage line length: %f" % (sum(line_lengths) / len(line_lengths)))
        print("\tMin line length: %d" % min(line_lengths))
        print("\tMax line length: %d" % max(line_lengths))
    return


if __name__ == '__main__':

    from gensim.corpora import WikiCorpus
    from gensim.models.keyedvectors import KeyedVectors

    cfg = Config()
    #w2v = KeyedVectors.load_word2vec_format(cfg['w2v_path'], binary=True)
    #wiki = WikiCorpus.load(cfg['wiki_path'])

    preprocess_brown(cfg['corpus__min_word_freq'], 5, cfg['corpus__data_file'],
                     cfg['corpus__dict_file'])
    split_dataset(cfg['corpus__data_file'], cfg['corpus__train_fraction'], cfg['corpus__test_fraction'], True)

    #preprocessWiki(wiki, w2v.vocab, w2v.syn0.shape[0], cfg['lm__input_file'])
