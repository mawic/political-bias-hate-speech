import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import tensorflow.keras.backend as K

from sklearn.model_selection import KFold
from helper.scoring import f1_score_overall
import gc

def build_word_index_for_fold(tweets):
    # A dictionary mapping words to an integer index
    word_index = build_words_indexing(tweets)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return(word_index, reverse_word_index)
  
def build_words_indexing(tweets, most_common=None):
    
    words = {}
    for index, tw in enumerate(tweets):
        for w in tw:
            try:
                words[w] +=1
            except:
                words[w] = 1    
                
    words_indexing = {}
    for index, word in enumerate(words):
        words_indexing[word] = index        

    words_indexing = {k:(v+2) for k,v in words_indexing.items()}
    words_indexing["<PAD>"] = 0
    words_indexing["<UNK>"] = 1
        
    return(words_indexing)

def convert_to_integers(sentence, words_indexing):
    
    integer_sentence = []
    for word in sentence:
        if word in words_indexing:
            integer_sentence.append(words_indexing[word])
        else:
            integer_sentence.append(words_indexing["<UNK>"])
    
    return(np.array(integer_sentence, dtype=int))

def create_train_test_folds(x_train, x_test, word_index):
    
    train_data = keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=30)

    test_data = keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=30)
    
    return(train_data, test_data)

def create_model(word_index):
    model = tf.keras.Sequential()
    model.add(keras.layers.Embedding(len(word_index), 50, mask_zero=True))
    model.add(Bidirectional(LSTM(64)))
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return(model)




def perform_cross_validation(folds=5,
                             tweets=None,
                             labels=None, 
                             print_fold_eval=True):
    fold_index = 0

    scores_clf = []
    
    for train_index, test_index in KFold(n_splits=folds, 
                                     shuffle=True, 
                                     random_state=123).split(tweets):


        fold_index = fold_index+1

        x_train, x_test = tweets[train_index],tweets[test_index]
        y_train, y_test = labels[train_index],labels[test_index]

        word_index, reverse_word_index = build_word_index_for_fold(x_train)    
        x_train = np.array([convert_to_integers(tweet, word_index) for tweet in x_train])
        x_test = np.array([convert_to_integers(tweet, word_index) for tweet in x_test])    

        train_data, test_data = create_train_test_folds(x_train, x_test, word_index)

        clf = create_model(word_index)
        
        clf_history = clf.fit(train_data,
                              y_train,
                              epochs=1,
                              batch_size=32,
                              verbose=0)

        score_clf = f1_score_overall(y_test, clf.predict(test_data), only_overall=False)
        

        scores_clf.append(score_clf)
        
        if(print_fold_eval):
            print("F1 Score - Fold {} : Overall: {}, None: {}, Offensive: {}".format(fold_index,scores_clf[-1][1],
                                                                                    scores_clf[-1][0][0],
                                                                                    scores_clf[-1][0][1]))

        if(fold_index == folds):
            score_none = []
            score_offensive = []
            score_overall = []

            for score in scores_clf:
                score_none.append(score[0][0])
                score_offensive.append(score[0][1])
                score_overall.append(score[1])


            print("Overall F1 Scores:\n None: {}\n Offensive:{}\n Overall:{}"\
                      .format((sum(score_none) / len(score_none)), (sum(score_offensive) / len(score_offensive)),\
                              (sum(score_overall) / len(score_overall))))
            
        tf.keras.backend.clear_session()
        del clf,clf_history
        
        
    return(scores_clf)