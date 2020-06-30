import pandas as pd
import csv

def get_all_germEval_data():
    #load dataset germEval 2019
    #emojies are not decoded properly
    germeval2019_subtask1_2_train = pd.read_csv('data/germEval2019/germeval2019.training_subtask1_2_korrigiert.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])
    #emojies are not decoded properly
    germeval2019_subtask1_2_test = pd.read_csv('data/germEval2019/germeval2019GoldLabelsSubtask1_2.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])

    #load dataset germEval 2018
    germEval2018_train = pd.read_csv('data/germEval2018/germeval2018.training.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])

    germEval2018_test = pd.read_csv('data/germEval2018/germeval2018.test.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])

    df = pd.concat([germeval2019_subtask1_2_train,
                   germeval2019_subtask1_2_test,
                   germEval2018_train,
                   germEval2018_test])

    #remove duplicate tweets, due concatinating different datasets together
    df = df.drop_duplicates()
    df = df.sample(frac=1,random_state=1993).reset_index(drop=True)

    return(df)


def get_train_and_test_data_germEval():
    #load dataset germEval 2019
    #emojies are not decoded properly
    germeval2019_subtask1_2_train = pd.read_csv('data/germEval2019/germeval2019.training_subtask1_2_korrigiert.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])
    #emojies are not decoded properly
    germeval2019_subtask1_2_test = pd.read_csv('data/germEval2019/germeval2019GoldLabelsSubtask1_2.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])

    #load dataset germEval 2018
    germEval2018_train = pd.read_csv('data/germEval2018/germeval2018.training.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])

    germEval2018_test = pd.read_csv('data/germEval2018/germeval2018.test.txt',
                    sep = "\t",encoding="utf-8",quoting=csv.QUOTE_NONE ,
                    names=['text','label_1','label_2'])

    train = pd.concat([germeval2019_subtask1_2_train,
                       germEval2018_train,
                       germEval2018_test])
    
    test = germeval2019_subtask1_2_test

    #remove duplicate tweets, due concatinating different datasets together
    train = train.drop_duplicates()
    train = train.sample(frac=1,random_state=1993).reset_index(drop=True)

    return(train,test)