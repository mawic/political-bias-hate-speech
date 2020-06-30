import re
from helper.preprocessing import *
import subprocess

def calculate_npmi(topics, name_reference_data,topn=10):    
    with open("helper/topic_interpretability/data/topics_germEval.txt", "w", encoding="utf8") as f:        
        for topic in topics:
            f.write(" ".join(topic)+"\n")
            
            
    metric="npmi" #evaluation metric: pmi, npmi or lcp

    topic_file="helper/topic_interpretability/data/topics_germEval.txt"
    ref_corpus_dir="helper/topic_interpretability/ref_corpus/"+name_reference_data#germEval"#germanTweets

    wordcount_file="helper/topic_interpretability/wordcount/wc-oc.txt"
    oc_file="helper/topic_interpretability/results/topics-oc.txt"
    
    
    subprocess.check_output("/usr/bin/python helper/topic_interpretability/ComputeWordCount.py "+topic_file+" "+ref_corpus_dir+" > "+wordcount_file, shell=True)
    
    result = subprocess.check_output("/usr/bin/python helper/topic_interpretability/ComputeObservedCoherence.py "+topic_file+" "+metric+" "+wordcount_file + " -t "+str(topn), shell=True)

    result = float(re.findall(r"Average Topic Coherence = (-?\d+\.\d+)", str(result))[0])

    
    return(result)

def create_dict(data, no_below, no_above, keep_n):
    
    dictionary = gensim.corpora.Dictionary(data)
    
    if((no_below!=None) and (no_above!=None) and (keep_n!=None)): 
        #filtered_dict = gensim.corpora.Dictionary(data)
        #filtered_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        
        #filtered_corpus = [filtered_dict.doc2bow(text) for text in data]
        print("DEPRECATED")
    else:
        filtered_dict = None
        filtered_corpus = None
    
    corpus = [dictionary.doc2bow(text) for text in data]
    
    return(dictionary, corpus, filtered_dict, filtered_corpus)


def get_LDA_topics(model,k,num_words):
    x = model.show_topics(num_topics=k, num_words=num_words,formatted=False)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

    topics_tokens = []
    for topic_id,words in topics_words:
        topics_tokens.append(words)

    topics = [" ".join(tops) for tops in topics_tokens]
    return(topics_tokens)
    

def csv_writer_generator(csv_fname, row):
    with open(csv_fname, "a", encoding="utf8") as f:
        f.write(" ".join(row)+"\n")
        
def write_new_train_test_data(train,test):   
        
    train_file = "helper/topic_interpretability/ref_corpus/germEval_train/corpus.0"
    test_file = "helper/topic_interpretability/ref_corpus/germEval_test/corpus.0"
    try:
        subprocess.check_output("rm "+train_file, shell=True)
        subprocess.check_output("rm "+test_file, shell=True)
    except:
        print("")
        
    for tw in train:
        csv_writer_generator(train_file,tw)

    for tw in test:
        csv_writer_generator(test_file,tw)

def filter_extremes(data, no_below, no_above, min_no_token):
    dict_words =  {}
    for tw in data:
        for token in tw:
            try:
                dict_words[token] += 1
            except:
                dict_words[token] = 1

    dict_words_filtered =  []
    for k, v in dict_words.items():
        if v >= no_below and v < int(len(dict_words)*no_above):
            dict_words_filtered.append(k)

    data_new=[]
    for tw in data:
        new_tw = []
        for token in tw:
            if token in dict_words_filtered:
                new_tw.append(token)
        if len(new_tw) >= min_no_token:
            data_new.append(new_tw)

    return(data_new)

def create_poooling_data(df_fresh,getStatsBack=False):

    hashtag_user = {}

    for tw in df_fresh.text:
        hashtags = re.findall("(#[a-zA-Z0-9_]+)",tw.lower())
        users = re.findall("(@[a-zA-Z0-9_]+)",tw.lower())

        for hashtag in hashtags:
            try:
                hashtag_user[hashtag] += 1
            except:
                hashtag_user[hashtag] = 1

    hashtag_user_filtered = {}

    for k,v in enumerate(hashtag_user):
        if(hashtag_user[v] > 1):
            hashtag_user_filtered[v] = []

    single_tw = 0
    for tw in df_fresh.text:
        hashtags = re.findall("(#[a-zA-Z0-9_]+)",tw.lower())
        users = re.findall("(@[a-zA-Z0-9_]+)",tw.lower())

        no_hashtag_match, no_user_match = False, False,

        if len(hashtags)>0:

            no_hastags_inner = 0
            for hashtag in hashtags:
                if hashtag in hashtag_user_filtered:
                    hashtag_user_filtered[hashtag].append(tw)
                else:
                    no_hastags_inner +=1

            if no_hastags_inner == len(hashtags):
                no_hashtag_match = True
        else:
            no_hashtag_match = True

        if((no_hashtag_match)):# 
            single_tw += 1
            hashtag_user_filtered[tw] = [tw]


    data = []
    for k,v in enumerate(hashtag_user_filtered):
        if len(hashtag_user_filtered[v])>1:
            data.append([v," . ".join(hashtag_user_filtered[v])])
        else:
            data.append([v,hashtag_user_filtered[v][0]])

    counter = 0
    for k,v in enumerate(hashtag_user_filtered):
        for tt in hashtag_user_filtered[v]:
            counter += 1
    
    if getStatsBack:
        return(data,hashtag_user,hashtag_user_filtered,single_tw)
    else:
        return(data)
        
def runLDA(dict_train, corpus_train, k,random_state=100):
    
    return(gensim.models.ldamodel.LdaModel(corpus=corpus_train,
                                           id2word=dict_train,
                                           num_topics=k, 
                                           random_state=random_state,
                                           update_every=0,
                                           passes=10,  #epochs
                                           iterations=200, #how many iterations the VB is allowed in the E-step/inference without convergence
                                           chunksize=10000,
                                           eval_every=None,
                                           alpha='auto',#'asymmetric',
                                           per_word_topics=True,
                                           #callbacks=callbacks
                                          ))

def get_topic_distribution(model, other_corpus):
    inferences = model.get_document_topics(other_corpus, minimum_probability=0.0)
    
    infereces_topics=[]
    for tw in inferences:
        topics = []
        for topic in tw:
            topics.append(topic[1])
        infereces_topics.append(topics)
        
    return(infereces_topics)

def assign_topic_distribution(df, model, dictionary):
    df = prepare_data(df)
    df_corpus = list(map(dictionary.doc2bow, df.token))
    df_topic_distribution = get_topic_distribution(model,df_corpus)
    df["topic_distribution"] = df_topic_distribution
    return(df)

def assign_topic_and_probability(df):
    df["topic"] = df.apply(lambda x: np.argmax(x["topic_distribution"]),axis=1)
    df["highest_probability"] = df.apply(lambda x: x["topic_distribution"][x["topic"]],axis=1)
    return(df)