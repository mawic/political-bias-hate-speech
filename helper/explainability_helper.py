import pickle
import tensorflow as tf

from helper.preprocessing import *
from helper.cross_val_model import *
from helper.classifier_helper import *

import shap

def get_dataset(data_pool, germEval_,germEval_topic_distribution):
        other = create_data_from_pool(data_pool, germEval_topic_distribution)    
        data = concat_germEval_and_Other(germEval_, other)
        data = prepare_data(data)    
        data = replace_label_to_binary(data)
        return(data)
    
def prepare_data(df):
    
    df["token"] = df.apply(lambda x: remove_hand_selected_words(x["text"]), axis=1)
    df["token"] = df.apply(lambda x: rermove_repeating_chars(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: emoji_2_text(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: " ".join(tw_process.pre_process_doc(x["token"])), axis=1)
    df["token"] = df.apply(lambda x: remove_special_chars(x["token"]), axis=1) #[^A-Za-z0-9\säüßöÖÄÜ<>_]
    df["token"] = df.apply(lambda x: x["token"].lower(), axis=1)
    df["token"] = df.apply(lambda x: remove_numbers(x["token"]), axis=1)

    df["token"] = df.apply(lambda x: sentence_to_token(x["token"]), axis=1)
    
    df["token"] = df.apply(lambda x: remove_stopwords(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: remove_ekphrasis_tokens(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: remove_emoji_special_tokens(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: remove_short_tokens(3, x["token"],list_=[]), axis=1)    
        
    return(df)

def create_model_for_explainability(word_index):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(len(word_index), 50))
    model.add(Bidirectional(LSTM(64)))
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    #model.summary()

    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return(model)

def create_train_label(data):
    tweets = np.array(data["token"].tolist())
    labels = np.array(data["label"].tolist())
    
    return(tweets,labels)

def prepare_data_for_training(tweets):
    word_index, reverse_word_index = build_word_index_for_fold(tweets)    
    x_train = np.array([convert_to_integers(tweet, word_index) for tweet in tweets])
    x_test = np.array([convert_to_integers(tweet, word_index) for tweet in tweets])    

    train_data, test_data = create_train_test_folds(x_train, x_test, word_index)
    return(train_data, test_data, word_index)

def train_model(tweets, label):
    
    train_data, test_data, word_index = prepare_data_for_training(tweets)

    clf = create_model_for_explainability(word_index)

    clf_history = clf.fit(train_data,
                          label,
                          epochs=1,
                          batch_size=32,
                          verbose=0)
    
    score_clf = f1_score_overall(label, clf.predict(test_data), only_overall=False)
    
    print("F1 Score - Train data: Overall: {}, None: {}, Offensive: {}".format(score_clf[1],
                                                                        score_clf[0][0],
                                                                        score_clf[0][1]))
    
    return(clf, word_index)


def create_train_test_folds(x_train, x_test, word_index):
    
    train_data = keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=30)

    test_data = keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=30)
    return(train_data,test_data)


def load_model_dict_data(path):
    with open(path+"_df", 'rb') as f:
        data_df = pickle.load(f)
    with open(path+"_dict", 'rb') as f:
        data_dict = pickle.load(f)
    model = tf.keras.models.load_model(path+"_model.h5")
    return(model, data_df, data_dict)

def prepare_data_for_training_single(tweets, word_index):
    x_train = np.array([convert_to_integers(tweet, word_index) for tweet in tweets])

    train_data, test_data = create_train_test_folds(x_train, x_train, word_index)
    return(train_data, test_data)


#SHAP
def shap_explain(model,x_test, words_indexing, explainer, shap_values, which_prediction=0):    
    print("which_prediction:",which_prediction)

    prediction = model.predict([x_test[which_prediction : which_prediction+1]])[0]

    if prediction > 0.5:
        predicted_class = 1
    else:
        predicted_class = 0
        
    print('{} : {}'.format(predicted_class, prediction[0]))

    words = words_indexing
    num2word = {}
    for w in words.keys():
        num2word[words[w]] = w
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_test[i]))) for i in range(10)])
    
    print(x_test_words[which_prediction])
    display(shap.force_plot(explainer.expected_value[0], shap_values[0][which_prediction], x_test_words[which_prediction]))
    

def do_explain_model(model, dict_, dataset, look_first=10 ):

    data = np.array([convert_to_integers(tweet, dict_) for tweet in dataset.token])
    data, data = create_train_test_folds(data, data, dict_)

    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data[:look_first])

    for i in range(0,10):
        shap_explain(model, data , dict_ ,explainer, shap_values, i)