import pandas as pd
import numpy as np
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import tensorflow as tf
from collections import Counter
from utilities import get_oov_vector
import nltk
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors
import tqdm
import os
import time
from preprocess_utils import Preprocessor
from augmentation import retranslation, mixup, synonyms
from architectures import BIRNN

#model_baseline = BIRNN.rnn_cnn
unknown_word = "_UNK_"
end_word = "_END_"
nan_word = "_NAN_"
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
results = pd.DataFrame(columns=['fold_id','epoch','roc_auc_v','roc_auc_t'])

train_data = pd.read_csv("assets/raw_data/train.csv")
test_data = pd.read_csv("assets/raw_data/test.csv")

sentences_train = train_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values

class Config:

    max_sentence_len = 500
    do_augmentation_with_translate = False
    do_augmentation_with_mixup = False
    do_synthezize_embeddings = False
    if do_synthezize_embeddings:
        synth_threshold = 0.1
    bsize = 256
    max_seq_len = 500
    epochs = 12
    model_name = 'caps_first_test'
    root = ''
    fp = 'models/CAPS/' + model_name + '/'
    logs_path = fp + 'logs/'
    if not os.path.exists(root + fp):
        os.mkdir(root + fp)

class ToxicComments:

    def __init__(self,Config):
        self.preprocessor = Preprocessor()
        self.cfg = Config()
        self.word_counter = Counter()
        self.words_dict = {}

    def tokenize_sentences(self,sentences, words_dict, mode = 'twitter'):
        twitter_tokenizer = TweetTokenizer()
        tokenized_sentences = []
        for sentence in tqdm.tqdm(sentences,mininterval=5):
            if hasattr(sentence, "decode"):
                sentence = sentence.decode("utf-8")
            sentence = self.preprocessor.expand_contractions(sentence)
            if mode == 'nltk':
                tokens = nltk.tokenize.word_tokenize(sentence)
            elif mode == 'twitter':
                tokens = twitter_tokenizer.tokenize(sentence)
            else:
                tokens = None
            result = []
            self.word_counter.update(tokens)
            for word in tokens:
                self.word_counter.update([word])
                word = word.lower()
                if word not in words_dict:
                    words_dict[word] = len(words_dict)
                result.append(word)
            tokenized_sentences.append(result)
        return tokenized_sentences, words_dict

    def tokenized_sentences2seq(self,tokenized_sentences, words_dict):
        sequences = []
        for sentence in tqdm.tqdm(tokenized_sentences, mininterval=5):
            seq = []
            for token in sentence:
                seq.append(words_dict[token])
            sequences.append(seq)
        return sequences

    def update_words_dict(self,tokenized_sentences):
        self.words_dict.pop(unknown_word, None)
        k = 0
        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in self.words_dict:
                    k += 1
                    self.words_dict[token] = len(self.words_dict)
        print('{} words added'.format(k))
        self.words_dict[unknown_word] = len(self.words_dict)
        self.id2word = dict((id, word) for word, id in self.words_dict.items())

    def clear_embedding_list(self,model, embedding_word_dict, words_dict):
        cleared_embedding_list = []
        cleared_embedding_word_dict = {}
        k = 0
        l = 0
        for word in tqdm.tqdm(words_dict):
            if word not in embedding_word_dict:
                if self.cfg.do_synthezize_embeddings:
                    l += 1
                    row = get_oov_vector(word, model, threshold=self.cfg.synth_threshold)
                    if row is None:
                        k += 1
                        continue
                    else:
                        cleared_embedding_list.append(row)
                        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
                else:
                    continue
            else:
                row = model[word]
                cleared_embedding_list.append(row)
                cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
        print('embeddings not found: {0:.1f}%'.format(l / len(words_dict) * 100))
        print('embeddings not synthesized: {0:.1f}%'.format(k / len(words_dict) * 100))
        return cleared_embedding_list, cleared_embedding_word_dict

    #def get_bad_sentences(self,vlosses, vlogits, X_valid, Y_valid):
    #    idx = (-vlosses).argsort()[:100]
    #    X = X_valid[idx]
    #    Y = Y_valid[idx]
    #    preds = np.concatenate((Y,vlogits[idx]))
    #    losses = vlosses[idx]
    #    sentences = []
    #    for row in X:
    #        sentences.append(' '.join([id_to_embedded_word[r] for r in row]))
    #    d = pd.DataFrame(preds, columns=list_classes.extend(['l' + label for label in list_classes]))
    #    #d[list_classes] = Y
    #    d['words'] = pd.Series(sentences)
    #    d['idx'] = pd.Series(idx)
    #    d['loss'] = pd.Series(losses)
    #    d.to_csv('misclassifies2.csv', index=False)

    def convert_tokens_to_ids(self,tokenized_sentences, embedding_word_dict):
        words_train = []

        for sentence in tqdm.tqdm(tokenized_sentences):
            current_words = []
            for word_index in sentence:
                word = self.id2word[word_index]
                word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
                current_words.append(word_id)

            if len(current_words) >= self.cfg.max_sentence_len:
                current_words = current_words[:self.cfg.max_sentence_len]
            else:
                current_words += [len(embedding_word_dict) - 1] * (self.cfg.max_sentence_len - len(current_words))
            words_train.append(current_words)
        return words_train

    def prepare_embeddings(self, words_dict, mode = 'fasttext_300d'):
        print("Loading embeddings...")

        if mode == 'fasttext_300d':
            model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/crawl-300d-2M.vec', binary=False)
        elif mode == 'mini_fasttext_300d':
            model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/mini_fasttext_300d2.vec',binary=False)

        else:
            model = None
        embedding_word_dict = {w:ind for ind,w in enumerate(model.index2word)}
        embedding_size = 300

        print("Preparing data...")
        embedding_list, embedding_word_dict = self.clear_embedding_list(model, embedding_word_dict, words_dict)

        del model

        embedding_word_dict[unknown_word] = len(embedding_word_dict)
        embedding_list.append([0.] * embedding_size)
        embedding_word_dict[end_word] = len(embedding_word_dict)
        embedding_list.append([-1.] * embedding_size)

        embedding_matrix = np.array(embedding_list)


        id_to_embedded_word = dict((id, word) for word, id in embedding_word_dict.items())
        return embedding_matrix, embedding_word_dict, id_to_embedded_word

    def fit_tokenizer(self,list_of_sentences):

        list_of_tokenized_sentences = []
        for sentences in list_of_sentences:
            tokenized_sentences, self.words_dict = self.tokenize_sentences(sentences, self.words_dict)
            list_of_tokenized_sentences.append(tokenized_sentences)

        self.words_dict[unknown_word] = len(self.words_dict)
        self.id2word = dict((id, word) for word, id in self.words_dict.items())

        return list_of_tokenized_sentences


cfg = Config()

tc = ToxicComments(Config)

Y = train_data[list_classes].values

tokenized_sentences_train, tokenized_sentences_test = tc.fit_tokenizer([sentences_train, sentences_test])

sequences_train = tc.tokenized_sentences2seq(tokenized_sentences_train, tc.words_dict)
sequences_test = tc.tokenized_sentences2seq(tokenized_sentences_test, tc.words_dict)
embedding_matrix, embedding_word_dict, id_to_embedded_word = tc.prepare_embeddings(tc.words_dict)

train_list_of_token_ids = tc.convert_tokens_to_ids(sequences_train, embedding_word_dict)
test_list_of_token_ids = tc.convert_tokens_to_ids(sequences_test, embedding_word_dict)

X = np.array(train_list_of_token_ids)
X_test = np.array(test_list_of_token_ids)

fold_size = len(X) // 10

fold_start = 0
fold_end = fold_start + fold_size

fold_id = 0
X_valid = X[fold_start:fold_end]
Y_valid = Y[fold_start:fold_end]
X_train = np.concatenate([X[:fold_start], X[fold_end:]])
Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])


def squash(vector, epsilon=1e-9):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''

    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)

def routing(input, b_IJ, iter_routing=3):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    bsize = input.get_shape()[0]
    num_caps = input.get_shape()[1]
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, num_caps, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [bsize, 1, 1, 1, 1])
    assert input.get_shape() == [bsize, num_caps, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [bsize, num_caps, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(3):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [bsize, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [bsize, 1, 10, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, num_caps, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [bsize, num_caps, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v
    return (v_J)





bsize = cfg.bsize

graph = tf.Graph()

with graph.as_default():
    tf.set_random_seed(1)

    x = tf.placeholder(tf.int32, shape=(None, cfg.max_seq_len), name="input_x")
    y = tf.placeholder(tf.float32, shape=(None, 6), name="input_y")
    keep_prob = tf.placeholder(dtype=tf.float32, name="input_keep_prob")

    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,
                                    initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    with tf.variable_scope('RNN_layer'):
        with tf.variable_scope('fw'):
            fw_cell1 = tf.nn.rnn_cell.GRUCell(64)
            fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)

        with tf.variable_scope('bw'):
            bw_cell1 = tf.nn.rnn_cell.GRUCell(64)
            bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell1, bw_cell1, embedded_input, dtype=tf.float32)
        output_fw, output_bw = outputs

        outputs = tf.concat([output_fw, output_bw], axis=2)

        outputs = tf.transpose(outputs, [0, 2, 1])

    # Primary Capsules layer, return [batch_size, ? , 8, 1]
    with tf.variable_scope('PrimaryCaps_layer'):
        capsules = tf.layers.conv1d(outputs, filters=32 * 8,
                                    kernel_size=9,
                                    strides=2,
                                    activation=tf.nn.relu)


        capsules = tf.expand_dims(capsules, 3)
        capsules = tf.reshape(capsules, (bsize, 60 * 32, 8, 1))
        capsules = squash(capsules)


    with tf.variable_scope('FCCaps_layer'):
        # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
        # caps2 = digitCaps(capsules_squashed)

        input = tf.reshape(capsules, shape=(bsize, 60 * 32, 1, 8, 1))

        with tf.variable_scope('routing'):
            # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
            # about the reason of using 'batch_size', see issue #21
            b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
            capsules = routing(input, b_IJ)
            capsules = tf.squeeze(capsules, axis=1)

        flat_capsules = tf.layers.flatten(capsules)

        logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)

        loss = binary_crossentropy(y, logits)
        cost = tf.losses.log_loss(predictions=logits, labels=y)
        # loss = tf.losses.sigmoid_cross_entropy(y,logits)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
        (_, auc_update_op) = tf.metrics.auc(
            predictions=logits,
            labels=y,
            curve='ROC')

        saver = tf.train.Saver(max_to_keep=15)


def save(sess, fold_id, epoch,roc_auc_valid, roc_auc_train):
    print('saving model...', end='')
    model_name = 'k%s_e%s.ckpt' % (fold_id,epoch)
    s_path = saver.save(sess, cfg.logs_path + model_name)
    print("Model saved in file: %s" % s_path)
    results.loc[len(results)] = [fold_id, epoch, roc_auc_valid, roc_auc_train]
    results.to_csv(cfg.fp + 'results.csv')


train_iters = len(X_train) - 2 * cfg.bsize
steps = train_iters // cfg.bsize
valid_iters = len(X_valid) -2 *  cfg.bsize

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(cfg.epochs):
        tic = time.time()
        costs = []
        step = 0
        tf.local_variables_initializer().run(session=sess)
        while step * cfg.bsize < train_iters:
            batch_x = X_train[step * cfg.bsize:(step + 1) * cfg.bsize]
            batch_y = Y_train[step * cfg.bsize:(step + 1) * cfg.bsize]
            cost_, _, roc_auc_train = sess.run([cost, optimizer, auc_update_op],
                                               feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: 0.7})
            if step % 10 == 0:
                print('e %s/%s  --  s %s/%s  -- cost %s' % (epoch, cfg.epochs, step, steps, cost_))
            costs.append(cost_)
            step += 1

        vstep = 0
        vcosts = []
        vlosses = np.asarray([])
        tf.local_variables_initializer().run(session=sess)
        while vstep * cfg.bsize < valid_iters:
            batch_x_valid = X_valid[vstep * cfg.bsize:(vstep + 1) * cfg.bsize]
            batch_y_valid = Y_valid[vstep * cfg.bsize:(vstep + 1) * cfg.bsize]
            test_cost_, valid_loss, roc_auc_valid = sess.run([cost, loss, auc_update_op],
                                                            feed_dict={x: batch_x_valid,
                                                                       y: batch_y_valid,
                                                                       keep_prob: 1
                                                                       })
            vstep += 1
            vcosts.append(test_cost_)
            vlosses = np.concatenate((vlosses, valid_loss))
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        toc = time.time()
        print('time needed %s' % (toc - tic))
        print('valid loss: %s' % avg_cost)
        print('roc auc test : {:.4}'.format(roc_auc_valid))
        print('roc auc train : {:.4}'.format(roc_auc_train))
        avg_train_cost = np.log(np.mean(np.exp(costs[:valid_iters])))
        print('train loss %s' % avg_train_cost)

        save(sess, fold_id, epoch,roc_auc_valid,roc_auc_train)













