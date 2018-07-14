"""
Exploration of mixup
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47730
https://arxiv.org/abs/1710.09412

"""

import numpy as np
import tqdm
import pandas as pd
from sklearn.utils import shuffle
import pickle
import random


def mixup(X_train, Y_train,alpha, portion, seed):

    np.random.seed(seed)
    indices = [ind for ind, x in enumerate(Y_train)]
    indices = np.random.permutation(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    return X_train, Y_train

