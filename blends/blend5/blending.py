# All credits goes to original authors.. Just another blend...
import pandas as pd
from sklearn.preprocessing import minmax_scale
import numpy as np
from global_variables import LIST_CLASSES

gru_public = pd.read_csv('models/PUBLIC/pooled-gru-fasttext/submission.csv') # PL score 0.9829
#lstm_nb_svm = pd.read_csv('models/PUBLIC/minimal-lstm-nb-svm-baseline-ensemble/submission.csv') # 0.9811
lstm = pd.read_csv('models/PUBLIC/improved-lstm-baseline-glove-dropout/submission.csv')
nb_svm = pd.read_csv('models/PUBLIC/nb-svm-strong-linear-baseline/submission.csv') #0.9771
lr = pd.read_csv('models/PUBLIC/logistic-regression-with-words-and-char-n-grams/submission.csv') # 0.9788
lgb_public = pd.read_csv('models/PUBLIC/lightgbm-with-select-k-best-on-tfidf/lgb_submission.csv') # 0.9785
lgb = pd.read_csv('models/LGB/hurford2/l2_test_data.csv') #0.9802
sup = pd.read_csv('models/PUBLIC/blend-of-blends-1/superblend_1.csv')
gru_ATT_4 = pd.read_csv('models/RNN/gru_ATT_4/l2_test_data.csv') #0.9848
gru_ATT_6_glove = pd.read_csv('models/RNN/gru_ATT_6_glove/l2_test_data.csv')
inception5b = pd.read_csv('models/CNN/inception5b/l2_test_data.csv')

for label in LIST_CLASSES:
    print('Scaling {}... Please stand by.'.format(label))
    lgb_public[label] = minmax_scale(lgb_public[label])
    gru_public[label] = minmax_scale(gru_public[label])
    lr[label] = minmax_scale(lr[label])
    lstm[label] = minmax_scale(lstm[label])
    nb_svm[label] = minmax_scale(nb_svm[label])
    gru_ATT_4[label] = minmax_scale(gru_ATT_4[label])
    gru_ATT_6_glove[label] = minmax_scale(gru_ATT_6_glove[label])
    inception5b[label] = minmax_scale(inception5b[label])
    sup[label] = minmax_scale(sup[label])


for label in LIST_CLASSES:
    print(label)
    print(np.corrcoef([lgb_public[label], gru_public[label], lr[label], lstm[label],nb_svm[label]]))


######
# l0   1/

blend_l0_1 = pd.DataFrame()
blend_l0_1['id'] = lgb_public['id']
blend_l0_1['toxic'] = lgb_public['toxic'] * 0.15 + gru_public['toxic'] * 0.4 + lr['toxic'] * 0.15 + lstm['toxic'] * 0.15 +nb_svm['toxic'] * 0.15
blend_l0_1['severe_toxic'] = lgb_public['severe_toxic'] * 0.15 + gru_public['severe_toxic'] * 0.4 + lr['severe_toxic'] * 0.15 + lstm['severe_toxic'] * 0.15 + nb_svm['severe_toxic'] * 0.15
blend_l0_1['obscene'] = lgb_public['obscene'] * 0.15 + gru_public['obscene'] * 0.4 + lr['obscene'] * 0.15 + lstm['obscene'] * 0.15 + nb_svm['obscene'] * 0.15
blend_l0_1['threat'] = lgb_public['threat'] * 0.15 + gru_public['threat'] * 0.4 + lr['threat'] * 0.15 + lstm['threat'] * 0.15 + nb_svm['threat'] * 0.15
blend_l0_1['insult'] = lgb_public['insult'] * 0.15 + gru_public['insult'] * 0.4 + lr['insult'] * 0.15 + lstm['insult'] * 0.15 + nb_svm['insult'] * 0.15
blend_l0_1['identity_hate'] = lgb_public['identity_hate'] * 0.15 + gru_public['identity_hate'] * 0.4 + lr['identity_hate'] * 0.15 + lstm['identity_hate'] * 0.15 + nb_svm['identity_hate'] * 0.15


################


blend_l1_1 = lgb_public.copy()
# keeping weight of single best model higher than other blends..
blend_l1_1[LIST_CLASSES] = 0.2 * minmax_scale(blend_l0_1[LIST_CLASSES].values) + 0.6 * gru_ATT_4[LIST_CLASSES].values + 0.2 * sup[LIST_CLASSES].values
blend_l1_1.to_csv("blends/blend3/submission.csv", index=False)


