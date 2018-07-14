from sklearn.model_selection import train_test_split
from toxic_comments.used_cleaned.preprocess_utils import preprocess
from toxic_comments.used_cleaned.global_variables import ASSET_PATH
import pandas as pd
import numpy as np
from toxic_comments.used_cleaned.augmentation import mixup
from toxic_comments.used_cleaned.architectures import BIRNN
import os
import tensorflow as tf
model_baseline = BIRNN().gru64_4


class Config():

    do_preprocess = True
    mixup_in_k_folds = True

    model_name = 'gru64_4'
    root = ''
    fp = 'TextClassification/models/' + model_name + '/'
    logs_path = fp + 'logs/'
    if not os.path.exists(root + fp):
        os.mkdir(root + fp)

class Model:

    def __init__(self, Config):
        self.cfg = Config
        self.graph = tf.Graph()

    def write_config(self):
        with open(os.path.join(self.cfg.fp, 'config.txt'), 'w') as f:
            f.write('Baseline = {}\n'.format(model_baseline.__name__))
            f.write('\n')
            f.write('Config\n')
            for line in self.class2list(self.cfg):
                f.write('{} = {}\n'.format(line[0], line[1]))

    @staticmethod
    def class2list(class_):
        class_list = [[item,class_.__dict__ [item]]for item in sorted(class_.__dict__ ) if not item.startswith('__')]
        return class_list

    def set_graph(self, embedding_matrix):
        pass

    def save(self, sess, fold_id, epoch,roc_auc_valid,roc_auc_train,cost_val):
        pass

    def train(self, X_train, Y_train, X_valid, Y_valid, X_test, embedding_matrix, fold_id=0, do_submission = False):

        self.write_config()

        ##
        ## hier statt tensorflow --> keras verwenden
        ##

        pass

def train_folds(fold_count=10):

    train_data = pd.read_csv(ASSET_PATH + "/train_data_firstPage.pickle")

    cfg = Config()
#    tc = ToxicComments(cfg)

    if cfg.do_preprocess:
        train_data = preprocess(train_data)

    sentences_train = train_data["comment_text"].fillna("_NAN_").values

    train_data, valid_data = train_test_split(train_data,test_size=0.2, random_state=123)
    valid_data, test_data =train_test_split(valid_data,test_size=0.5, random_state=123)

    Y = train_data["label"].values







    X = np.array((1,2,3))





    fold_size = len(X) // fold_count
    for fold_id in range(fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        X_valid = X[fold_start:fold_end]
        Y_valid = Y[fold_start:fold_end]
        X_train = np.concatenate([X[:fold_start], X[fold_end:]])
        Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

        if cfg.mixup_in_k_folds:
            X_train, Y_train = mixup( X_train, Y_train,0.5, 0.1, seed=43)

        m = Model(Config)
        m.set_graph(embedding_matrix)
        m.train(X_train, Y_train, X_valid, Y_valid, X_test, embedding_matrix, fold_id)









    if tc.cfg.level == 'word':
        tokenized_sentences_train, tokenized_sentences_valid,tokenized_sentences_test = tc.tokenize_list_of_sentences([sentences_train,sentences_valid,sentences_test])
        tokenized_sentences_train = [tc.preprocessor.rm_hyperlinks(s) for s in tokenized_sentences_train]
        tokenized_sentences_valid = [tc.preprocessor.rm_hyperlinks(s) for s in tokenized_sentences_valid]
        tokenized_sentences_test = [tc.preprocessor.rm_hyperlinks(s) for s in tokenized_sentences_test]

        tc.create_word2id([tokenized_sentences_train,tokenized_sentences_valid,tokenized_sentences_test])
        with open(tc.cfg.fp + 'tc_words_dict.p','wb') as f:
            pickle.dump(tc.word2id, f)

        sequences_train = tc.tokenized_sentences2seq(tokenized_sentences_train, tc.word2id)
        #sequences_test = tc.tokenized_sentences2seq(tokenized_sentences_test, tc.words_dict)
        if cfg.use_saved_embedding_matrix:
            with open(tc.cfg.fp + 'embedding_word_dict.p','rb') as f:
                embedding_word_dict = pickle.load(f)
            embedding_matrix = np.load(tc.cfg.fp + 'embedding.npy')
            id_to_embedded_word = dict((id, word) for word, id in embedding_word_dict.items())

        else:
            embedding_matrix, embedding_word_dict, id_to_embedded_word = tc.prepare_embeddings(tc.word2id)
            coverage(tokenized_sentences_train,embedding_word_dict)
            with open(tc.cfg.fp + 'embedding_word_dict.p','wb') as f:
                pickle.dump(embedding_word_dict,f)
            np.save(tc.cfg.fp + 'embedding.npy',embedding_matrix)

        train_list_of_token_ids = tc.convert_tokens_to_ids(sequences_train, embedding_word_dict)
        #test_list_of_token_ids = tc.convert_tokens_to_ids(sequences_test, embedding_word_dict)

        X = np.array(train_list_of_token_ids)
        #X_test = np.array(test_list_of_token_ids)
        X_test = None
    else:
        tc.preprocessor.min_count_chars = tc.cfg.min_count_chars

        tc.preprocessor.create_char_vocabulary(sentences_train)
        with open(tc.cfg.fp + 'char2index.p','wb') as f:
            pickle.dump(tc.preprocessor.char2index,f)

        X = tc.preprocessor.char2seq(sentences_train, maxlen=tc.cfg.max_seq_len)
        embedding_matrix = np.zeros((tc.preprocessor.char_vocab_size, tc.cfg.char_embedding_size))

        X_test = None




