import emoji
import numpy as np
import re
from langdetect import detect
import pandas as pd 

import gensim

from nltk.corpus import stopwords
stop_words_ger = stopwords.words('german')
stop_words_en = stopwords.words('english')

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

import spacy
nlp = spacy.load('de')

def get_text_processor():
    return(TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
        
        segmenter="twitter", 
        corrector="twitter", 
        
        fix_html=True,  # fix HTML tokens

        unpack_hashtags=False,  # perform word segmentation on hashtags
        unpack_contractions=False,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        tokenizer=SocialTokenizer(lowercase=False).tokenize,
        dicts=[emoticons]
    ))
tw_process = get_text_processor()


def prepare_data(data,do_lemmatize=False,do_ngram=False,onlyNounsAndNE=False,preparePoolingData=False):
    
    if preparePoolingData:
        df = pd.DataFrame(data,columns=['key',"text"])
    else:
        df = data.copy()

    df["token"] = df.apply(lambda x: remove_hand_selected_words(x["text"]), axis=1)
    df["token"] = df.apply(lambda x: rermove_repeating_chars(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: emoji_2_text(x["token"]), axis=1)
    df["token"] = df.apply(lambda x:annotate_usernames(x["token"]),axis=1)
    df["token"] = df.apply(lambda x: " ".join(tw_process.pre_process_doc(x["token"])), axis=1)
    df["token"] = df.apply(lambda x: remove_special_chars(x["token"]), axis=1) #[^A-Za-z0-9\säüßöÖÄÜ<>_]
    df["token"] = df.apply(lambda x: x["token"].lower(), axis=1)
    df["token"] = df.apply(lambda x: remove_numbers(x["token"]), axis=1)

    df["token"] = df.apply(lambda x: sentence_to_token(x["token"]), axis=1)
    
    df["token"] = df.apply(lambda x: remove_stopwords(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: remove_ekphrasis_tokens(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: remove_emoji_special_tokens(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: remove_short_tokens(3, x["token"],list_=[]), axis=1)    
    df["token"] = df.apply(lambda x: annotate_special_words(x["token"]), axis=1)
    df["token"] = df.apply(lambda x: remove_hand_selected_tokens(x["token"]), axis=1)
    
    if onlyNounsAndNE:
        df["token"] = df.apply(lambda x: keepOnlyNounsAndNE(x["token"]), axis=1)   

    if do_lemmatize:
        df["token"] = df.apply(lambda x: lemmatize_tokens(x["token"]), axis=1)
    
    if do_ngram:
        df.token = build_N_grams(df.token,10)    
        
    return(df)

def remove_numbers(s):
    return(re.sub(" \d+", " ", s))

def lemmatize_tokens(tokens):
    txt = " ".join(tokens)
    doc = nlp(txt)
    return([token.lemma_ for token in doc])

def keepOnlyNounsAndNE(tokens):
    txt = " ".join(tokens)
    doc = nlp(txt)
    
    keep = [str(w) for w in list(doc.ents)]
    for w in doc:
        if w.pos_ in ['NOUN']:
            keep.append(str(w.text))
    return([w for w in tokens if w in keep])

#load the bekannte parteien, medien, politiker, bundesländer, städte, bewegungen, sonstiges
with open("data/parteien_namem_etc.txt") as f:
    special_parteien_etc = [w.lower() for w in f.read().split("\n")]
    
    
def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def build_N_grams(text,threshold):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(text, min_count=5, threshold=threshold) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[text], threshold=threshold)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words_bigrams = make_bigrams(text, bigram_mod)
    return(data_words_bigrams)

def remove_stopwords(tweet):
    txt = []
    for w in tweet:
        if w in stop_words_ger:
            continue
            
        if w in stop_words_en:
            continue
        txt.append(w)

    return(txt)

def remove_short_tokens(min_length=2,text=None,list_ = ['.',',','?','!']):
    return([token for token in text if ((len(str(token).strip())>=min_length) or (token in list_))])

def remove_ekphrasis_tokens(tokens):
    return [token for token in tokens if '<' not in token]

def remove_emoji_special_tokens(tokens):
    return [token for token in tokens if 'fe0f'!=token and token !='face_with_head']

def annotate_usernames(text):
    tweet = text
    names = re.findall("(@[a-zA-Z0-9_]+)",tweet)
    names = list(set(names))
    mapper = {}
    for name in names:
        for special_name in special_parteien_etc:
            if special_name in name.lower():
                if((name in mapper.keys()) and (len(mapper[name])>0)):
                    mapper[name] = mapper[name] + " " + special_name
                else:
                    mapper[name] = special_name              
        if(name not in mapper.keys()):
            mapper[name] = ""            
    for k, v in mapper.items():
        tweet = tweet.replace(k, v)
    return(tweet)

def remove_hand_selected_words(text):
    return(re.sub(r"\|lbr\||\|LBR\||\|AMP\||&gt;|&amp;|\*innen|/sth|/fxn", " ", text))

def remove_hand_selected_tokens(tokens):
    list_ = ["uhr","ach"]#
    return([token for token in tokens if token not in list_])


def rermove_repeating_chars(text):
    to_remove = ".,?!"
    return(re.sub("(?P<char>[" + re.escape(to_remove) + "])(?P=char)+", r"\1", text))

from emoji import EMOJI_UNICODE, UNICODE_EMOJI
UNICODE_EMOJI = {v.encode('unicode-escape').decode().replace("\\", "").lower(): "<"+k.replace(":","").replace('-','')+">" for k, v in EMOJI_UNICODE.items()}

def emoji_2_text(text):
    
    text = emoji.demojize(text, delimiters=("<",">"))
    re_matches = re.findall(r"(<U\+[0-9a-zA-Z]*>)", text)
    
     
    for emoji_unicode in re_matches:
        try:
            text = text.replace(emoji_unicode, UNICODE_EMOJI[re.sub('[<>+]', '', emoji_unicode).lower()])
        except:
            text = text.replace(emoji_unicode,"")
    
        text = text.replace(emoji_unicode,"")
        
    m = re.search('<[a-z_-]*(-)[a-z_-]*>',text)
    if m is not None:
        old_emoji = m.group(0)
        new_emoji = m.group(0).replace('-','_')
        text = text.replace(old_emoji,new_emoji)
    return(text)

def remove_special_chars(text):
    return(re.sub(r"[^A-Za-z0-9\säüßöÖÄÜ<>_]+", " ", text))

def remove_token_w_len_1(text):
    tokens = text.split(" ")
    keep = [".","?","!",","]
    tw = []
    for tk in tokens:
        if((len(tk)>1) or (tk in keep)):
            tw.append(tk)
            
    return(" ".join(tw))

def detect_language(tw):
    try:
        return(detect(tw))
    except:
        return("unk")
    
def sentence_to_token(text):
    return([w.strip() for w in text.split()])

def annotate_special_words(tokens):
    dict_annotate = {
                'frauen': ['frauen',"frau"],
                'männer': ['mann','männer'],
                'links': ['links','linke','linken','linker','linkem'],
                'rechts': ['rechts','rechte','rechten','rechter',"rechtem"],
                'deutschland': ['ger','german','deutsche','deutsch','deutschen'],
                'nazi': ['nazi','nazis','neonazis'],
                'jude': ['juden'],
                'flüchtling': ['flüchtlinge',"fluechligne","fluechling"], 
                'grüne': ['gruene',"grünen","grüne"],     
                    }
    
    new_tw = []
    for token in tokens:
        new_token = None
        if token in dict_annotate['frauen']:
            new_token = 'frau'
        elif token in dict_annotate['männer']:
            new_token = 'mann'
        elif token in dict_annotate['links']:
            new_token = 'links'
        elif token in dict_annotate['rechts']:
            new_token = 'rechts'
        elif token in dict_annotate['deutschland']:
            new_token = 'deutschland'
        elif token in dict_annotate['nazi']:
            new_token = 'nazi'
        elif token in dict_annotate['jude']:
            new_token = 'jude'
        elif token in dict_annotate['flüchtling']:
            new_token = 'flüchtling'
        elif token in dict_annotate['grüne']:
            new_token = 'grüne'
        else:
            new_token=token
        new_tw.append(new_token)
        
    return(new_tw)
