

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


import keras
from keras.datasets import mnist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from IPython.display import clear_output
from sklearn.datasets import load_digits
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential, Model
from keras.layers import *
from sklearn import datasets
#Load the training and testing data
from sklearn.utils import shuffle

np.random.seed(1338)


def load_cora():

    from examples.utils import load_data, get_splits, preprocess_adj_numpy

    # Prepare Data
    X, A, Y = load_data(path='keras-deep-graph-learning/examples/data/cora/', dataset='cora')
    A = np.array(A.todense())

    _, Y_val, _, train_idx, val_idx, test_idx, train_mask = get_splits(Y)
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    labels = np.argmax(Y, axis=1) + 1

    # Normalize X
    X /= X.sum(1).reshape(-1, 1)
    X = np.array(X)

    Y_train = np.zeros(Y.shape)
    labels_train = np.zeros(labels.shape)
    Y_train[train_idx] = Y[train_idx]
    labels_train[train_idx] = labels[train_idx]

    Y_test = np.zeros(Y.shape)
    labels_test = np.zeros(labels.shape)
    Y_test[test_idx] = Y[test_idx]
    labels_test[test_idx] = labels[test_idx]

    # Build Graph Convolution filters
    SYM_NORM = True
    A_norm = preprocess_adj_numpy(A, SYM_NORM)
    return X, Y_train, Y_test, A, train_idx, val_idx, test_idx, train_mask


def load_mutag():

    from keras_dgl.layers import MultiGraphCNN
    from examples.utils import load_data, get_splits, preprocess_adj_numpy

    A_orig = pd.read_csv('keras-deep-graph-learning/examples/data/A_mutag.csv', header=None)
    A_orig = np.array(A_orig)
    orig_num_graph_nodes = A_orig.shape[1]
    orig_num_graphs = int(A_orig.shape[0] / A_orig.shape[1])


    A_orig = np.split(A_orig, orig_num_graphs, axis=0)
    A_orig = np.array(A_orig)

    # prepare data
    A = pd.read_csv('keras-deep-graph-learning/examples/data/A_edge_matrices_mutag.csv', header=None)
    A = np.array(A)


    num_graphs = 188  # hardcoded for mutag dataset

    A = np.split(A, num_graphs, axis=0)
    A = np.array(A)
    num_edge_features = int(A.shape[1]/A.shape[2])

    X = pd.read_csv('keras-deep-graph-learning/examples/data/X_mutag.csv', header=None)
    X = np.array(X)
    X = np.split(X, num_graphs, axis=0)
    X = np.array(X)

    num_graph_nodes = A.shape[1]
    num_graphs = int(A.shape[0] / A.shape[1])


    Y = pd.read_csv('keras-deep-graph-learning/examples/data/Y_mutag.csv', header=None)
    Y = np.array(Y)

    A, X, Y = shuffle(A, X, Y)
    return A, A_orig, X, Y, num_edge_features, num_graph_nodes, num_graphs, orig_num_graph_nodes, orig_num_graphs

def fix_gcn_paths():
    if 'keras-deep-graph-learning' not in os.getcwd(): sys.path.extend([os.path.join(os.getcwd(), 'keras-deep-graph-learning'), os.path.join(os.getcwd(), 'keras-deep-graph-learning/examples'), os.path.join(os.getcwd(), 'keras-deep-graph-learning/examples/data')])
    if '__init__' not in os.listdir('keras-deep-graph-learning/examples'): open('keras-deep-graph-learning/examples/__init__.py', 'w').write('')
    if '__init__' not in os.listdir('keras-deep-graph-learning/keras_dgl'): open('keras-deep-graph-learning/keras_dgl/__init__.py', 'w').write('')


def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Log-loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show()

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    batch_size = 128
    num_classes = 10
    epochs = 12
    if K.image_data_format() == 'channels_first': shape_ord = (1, img_rows, img_cols)
    else:  shape_ord = (img_rows, img_cols, 1)
    X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
    X_test = X_test.reshape((X_test.shape[0],) + shape_ord)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (X_train, y_train), (X_test, y_test)

def show_mnist_teaser():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()




def convert_sentence_to_token_mode1(sentence, vocab_path):
    import tokenization
    import numpy as np
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
    sentence = sentence.split('[MASK]')             
    tokens = ['[CLS]']                              
    for i in range(len(sentence)):
        if i == 0:
            tokens = tokens + tokenizer.tokenize(sentence[i]) 
        else:
            tokens = tokens + ['[MASK]'] + tokenizer.tokenize(sentence[i]) 
    tokens = tokens + ['[SEP]']                     
    token_input = tokenizer.convert_tokens_to_ids(tokens)        
    token_input = token_input + [0] * (512 - len(token_input))
    return tokens, token_input

def create_input_mask_mode1(token_input):
    mask_input = [0]*512
    for i in range(len(mask_input)):
        if token_input[i] == 103:
            mask_input[i] = 1
    return np.asarray([mask_input])


def create_phrase_mask_mode1():
    seg_input = [0]*512
    return np.asarray([seg_input])


def convert_sentence_to_token_mode2(sentence_1, sentence_2, vocab_path):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
    tokens_sen_1 = tokenizer.tokenize(sentence_1)
    tokens_sen_2 = tokenizer.tokenize(sentence_2)
    tokens = ['[CLS]'] + tokens_sen_1 + ['[SEP]'] + tokens_sen_2 + ['[SEP]']
    token_input = tokenizer.convert_tokens_to_ids(tokens)
    token_input = token_input + [0] * (512 - len(token_input))
    return token_input, tokens_sen_1, tokens_sen_2


def create_input_mask_mode2():
    mask_input = [0] * 512
    return np.asarray([mask_input])


def create_phrase_mask_mode2(tokens_sen_1, tokens_sen_2):
    seg_input = [0]*512
    len_1 = len(tokens_sen_1) + 2              
    for i in range(len(tokens_sen_2)+1):            
            seg_input[len_1 + i] = 1 
    return np.asarray([seg_input])



class TimestepDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape



    
class SampledSoftmax(Layer):
    def __init__(self, num_classes=50000, num_sampled=1000, tied_to=None, **kwargs):
        super(SampledSoftmax, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.tied_to = tied_to
        self.sampled = (self.num_classes != self.num_sampled)

    def build(self, input_shape):
        if self.tied_to is None:
            self.softmax_W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), name='W_soft', initializer='lecun_normal')
        self.softmax_b = self.add_weight(shape=(self.num_classes,), name='b_soft', initializer='zeros')
        self.built = True

    def call(self, x, mask=None):
        lstm_outputs, next_token_ids = x

        def sampled_softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            batch_losses = K.tf.nn.sampled_softmax_loss(
                self.softmax_W if self.tied_to is None else self.tied_to.weights[0], self.softmax_b,
                next_token_ids_batch, lstm_outputs_batch,
                num_classes=self.num_classes,
                num_sampled=self.num_sampled,
                partition_strategy='div')
            batch_losses = K.tf.reduce_mean(batch_losses)
            return [batch_losses, batch_losses]

        def softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            logits = K.tf.matmul(lstm_outputs_batch,
                                 K.tf.transpose(self.softmax_W) if self.tied_to is None else K.tf.transpose(self.tied_to.weights[0]))
            logits = K.tf.nn.bias_add(logits, self.softmax_b)
            batch_predictions = K.tf.nn.softmax(logits)
            labels_one_hot = K.tf.one_hot(K.tf.cast(next_token_ids_batch, dtype=K.tf.int32), self.num_classes)
            batch_losses = K.tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
            return [batch_losses, batch_predictions]

        losses, predictions = K.tf.map_fn(sampled_softmax if self.sampled else softmax, [lstm_outputs, next_token_ids])
        self.add_loss(0.5 * K.tf.reduce_mean(losses[0]))
        return lstm_outputs if self.sampled else predictions

    def compute_output_shape(self, input_shape):
        return input_shape[0] if self.sampled else (input_shape[0][0], input_shape[0][1], self.num_classes)
    
    
import numpy as np


class LMDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.indices)/self.batch_size))

    def __init__(self, corpus, vocab, sentence_maxlen=100, token_maxlen=50, batch_size=32, shuffle=True, token_encoding='word'):
        """Compiles a Language Model RNN based on the given parameters
        :param corpus: filename of corpus
        :param vocab: filename of vocabulary
        :param sentence_maxlen: max size of sentence
        :param token_maxlen: max size of token in characters
        :param batch_size: number of steps at each batch
        :param shuffle: True if shuffle at the end of each epoch
        :param token_encoding: Encoding of token, either 'word' index or 'char' indices
        :return: Nothing
        """

        self.corpus = corpus
        self.vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab).readlines()}
        self.sent_ids = corpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sentence_maxlen = sentence_maxlen
        self.token_maxlen = token_maxlen
        self.token_encoding = token_encoding
        with open(self.corpus) as fp:
            self.indices = np.arange(len(fp.readlines()))
            newlines = [index for index in range(0, len(self.indices), 2)]
            self.indices = np.delete(self.indices, newlines)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Read sample sequences
        word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
        if self.token_encoding == 'char':
            word_char_indices_batch = np.full((len(batch_indices), self.sentence_maxlen, self.token_maxlen), 260, dtype=np.int32)

        for i, batch_id in enumerate(batch_indices):
            # Read sentence (sample)
            word_indices_batch[i] = self.get_token_indices(sent_id=batch_id)
            if self.token_encoding == 'char':
                word_char_indices_batch[i] = self.get_token_char_indices(sent_id=batch_id)

        # Build forward targets
        for_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

        padding = np.zeros((1,), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch ):
            for_word_indices_batch[i] = np.concatenate((word_seq[1:], padding), axis=0)

        for_word_indices_batch = for_word_indices_batch[:, :, np.newaxis]

        # Build backward targets
        back_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch):
            back_word_indices_batch[i] = np.concatenate((padding, word_seq[:-1]), axis=0)

        back_word_indices_batch = back_word_indices_batch[:, :, np.newaxis]

        return [word_indices_batch if self.token_encoding == 'word' else word_char_indices_batch, for_word_indices_batch, back_word_indices_batch], []

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_token_indices(self, sent_id):
        sent_id = int(sent_id)
        with open(self.corpus) as fp:
            for i, line in enumerate(fp):
                if i == sent_id:
                    token_ids = np.zeros((self.sentence_maxlen,), dtype=np.int32)
                    # Add begin of sentence index
                    token_ids[0] = self.vocab['<bos>']
                    for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]):
                        if token.lower() in self.vocab:
                            token_ids[j + 1] = self.vocab[token.lower()]
                        else:
                            token_ids[j + 1] = self.vocab['<unk>']
                    # Add end of sentence index
                    if token_ids[1]:
                        token_ids[j + 2] = self.vocab['<eos>']
                    return token_ids

    def get_token_char_indices(self, sent_id):
        sent_id = int(sent_id)
        def convert_token_to_char_ids(token, token_maxlen):
            bos_char = 256  # <begin sentence>
            eos_char = 257  # <end sentence>
            bow_char = 258  # <begin word>
            eow_char = 259  # <end word>
            pad_char = 260  # <pad char>
            char_indices = np.full([token_maxlen], pad_char, dtype=np.int32)
            # Encode word to UTF-8 encoding
            word_encoded = token.encode('utf-8', 'ignore')[:(token_maxlen - 2)]
            # Set characters encodings
            # Add begin of word char index
            char_indices[0] = bow_char
            if token == '<bos>':
                char_indices[1] = bos_char
                k = 1
            elif token == '<eos>':
                char_indices[1] = eos_char
                k = 1
            else:
                # Add word char indices
                for k, chr_id in enumerate(word_encoded, start=1):
                    char_indices[k] = chr_id + 1
            # Add end of word char index
            char_indices[k + 1] = eow_char
            return char_indices

        with open(self.corpus) as fp:
            for i, line in enumerate(fp):
                if i == sent_id:
                    token_ids = np.zeros((self.sentence_maxlen, self.token_maxlen), dtype=np.int32)
                    # Add begin of sentence char indices
                    token_ids[0] = convert_token_to_char_ids('<bos>', self.token_maxlen)
                    # Add tokens' char indices
                    for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]):
                        token_ids[j + 1] = convert_token_to_char_ids(token, self.token_maxlen)
                    # Add end of sentence char indices
                    if token_ids[1]:
                        token_ids[j + 2] = convert_token_to_char_ids('<eos>', self.token_maxlen)
        return token_ids
    
parameters = {
    'multi_processing': False,
    'n_threads': 4,
    'cuDNN': True if len(K.tensorflow_backend._get_available_gpus()) else False,
    'train_dataset': 'wikitext-2/wiki.train.tokens',
    'valid_dataset': 'wikitext-2/wiki.valid.tokens',
    'test_dataset': 'wikitext-2/wiki.test.tokens',
    'vocab': 'wikitext-2/wiki.vocab',
    'vocab_size': 28914,
    'num_sampled': 1000,
    'charset_size': 262,
    'sentence_maxlen': 100,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 10,
    'patience': 2,
    'batch_size': 1,
    'clip_value': 1,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': True,
    'n_lstm_layers': 2,
    'n_highway_layers': 2,
    'cnn_filters': [[1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                    ],
    'lstm_units_size': 400,
    'hidden_units_size': 200,
    'char_embedding_size': 16,
    'dropout_rate': 0.1,
    'word_dropout_rate': 0.05,
    'weight_tying': True,
}
