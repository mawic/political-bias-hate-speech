import pandas as pd
import numpy as np
import pickle
from helper.preprocessing import *
from helper.cross_val_model import *

def create_data_from_pool(sample_pool, germEval_topics):
    tw = []

    for i in germEval_topics.index:
        tweets = sample_pool[i].sample(germEval_topics[i])
        tw.append(tweets)

    df = pd.concat(tw,ignore_index=True)
    df["label"] = "OTHER"
    df = df.filter(["tweet_id","text","label"])
    return(df)

def create_sample_pool(other, germEval_topics, sample_factor = 5):
    tweets = {}
    total_amount_of_tweets = 0
    
    try:
        print("Dataset:",other.orientation[0], ", sample_factor:",sample_factor)
    except:
        print("missing columns",-1)
        
    for i in germEval_topics.index:
        tweets_needed = germEval_topics[i]*sample_factor

        topic = other.loc[other["topic"]==i,].reset_index(drop=True)        
        tweets[i] = topic.sort_values(by="highest_probability",ascending=False)[:tweets_needed].reset_index(drop=True)
        
        total_amount_of_tweets += tweets[i].shape[0]
        
        try:
            print(i," Tweets: ",topic.shape[0], "\tSample pool:",tweets[i].shape[0]
                  ,"\tlowest prob:",round(tweets[i].tail(1).highest_probability.values[0],2),"\t"
                 ,"Proportion stream:",round(tweets[i].recording_type.value_counts()["stream"]/tweets[i].shape[0],2))
        except:
            print("missing columns",i)
            
    print("Total amount of tweets in pool:", total_amount_of_tweets,"\n")

    return(tweets)

def concat_germEval_and_Other(germEval_, other):
    germEval_ = germEval_.loc[germEval_["label"]=="OFFENSE",["text","label"]]
    other = other.filter(["text","label"])
    
    df = pd.concat([germEval_, other],ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return(df)

def replace_label_to_binary(df):
    df.loc[df.label == "OTHER","label"] = 0
    df.loc[df.label == "OFFENSE","label"] = 1
    return(df)

def add_tweet_id_to_list(data, tweetIds_used_in_last_experiment):
    counter = 0
    for idx in data.tweet_id.values:
        if idx not in  tweetIds_used_in_last_experiment:
            tweetIds_used_in_last_experiment.append(idx)
            counter += 1
    print("New Tweets in data: ", counter,"\n")    

def run_experiment_full_data_exchange(data_pool,
                                      germEval,
                                      germEval_topic_distribution,
                                      folds_per_run=5,
                                      runs=1):
    
    scores = []
    tweetIds_used_in_last_experiment = []
    
    for i in range(0,runs):
        
        print("\n#### Run:", i, "####\n")        

        other = create_data_from_pool(data_pool, germEval_topic_distribution)    
        data = concat_germEval_and_Other(germEval, other)
        data = prepare_data(data)    
        data = replace_label_to_binary(data)
        
        try:
            add_tweet_id_to_list(other, tweetIds_used_in_last_experiment)
        except:
            print("GermEval currently running")
        
        tweets = np.array(data["token"].tolist())
        labels = np.array(data["label"].tolist())

        scores_model = perform_cross_validation(folds=folds_per_run,
                                         tweets=tweets,
                                         labels=labels,
                                         print_fold_eval=True)
        
        scores.append(scores_model)
    return(scores, tweetIds_used_in_last_experiment)

def run_experiment_partly_data_exchange(data_pool,
                                      germEval,
                                      germEval_topic_distribution,
                                      folds_per_run=5,
                                      runs=1,
                                      steps_data_exchange=10):
    
    scores = []
    tweetIds_used_in_last_experiment = []
    
    for i in range(0,runs):
        
        print("\n#### Run:", i, "####\n")

        scores_data_exchange = []
        
        other = create_data_from_pool(data_pool, germEval_topic_distribution)   
        germEval_only_other = germEval.loc[germEval.label=="OTHER" , :]
        
        splits_germEval = np.array_split(germEval_only_other, steps_data_exchange)      
        splits_other= np.array_split(other, steps_data_exchange)
        
        germEval_index = []
        other_index = []
        
        try:
            add_tweet_id_to_list(other, tweetIds_used_in_last_experiment)
        except:
            print("GermEval currently running")
        
        for j in range(0,steps_data_exchange+1):
            
            if j == 0:
                df = germEval.copy()
            else:
                
                print("\nReplaced",j,"/",steps_data_exchange," of germEval data with label==OTHER\n")
                
                germEval_index.append(splits_germEval[j-1].index)
                other_index.append(splits_other[j-1].index)

                germEval_index_flattern = [item for sublist in germEval_index for item in sublist]
                other_index_flattern = [item for sublist in other_index for item in sublist]

                germEval_to_replace = germEval[germEval.index.isin(germEval_index_flattern)==False].copy()
                other_to_insert = other.iloc[other_index_flattern,:].copy()

                df = pd.concat([germEval_to_replace[["text","label"]], other_to_insert[["text","label"]]])
                df = df.sample(frac=1).reset_index(drop=True)                
                
                print("Size of old GermEval:",germEval_to_replace.loc[germEval_to_replace.label=="OTHER",:].shape[0])
                print("Size of other:",other_to_insert.shape[0],"\n")
                print("New Dataframe:")
                print(df.label.value_counts(),"\n")
            
            df = prepare_data(df)    
            df = replace_label_to_binary(df)

            tweets = np.array(df["token"].tolist())
            labels = np.array(df["label"].tolist())

            scores_model = perform_cross_validation(folds=folds_per_run,
                                             tweets=tweets,
                                             labels=labels,
                                             print_fold_eval=True)
        
            scores_data_exchange.append(scores_model)

        scores.append(scores_data_exchange)
        
    return(scores, tweetIds_used_in_last_experiment)

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

def get_topic_distribution_over_dataset(df):
    df = df.topic.value_counts().sort_index()
    return(df)
