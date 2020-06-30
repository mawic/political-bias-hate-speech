import numpy as np
import pickle
import statistics
from os import listdir
from os.path import isfile, join

def get_result_files(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))] 
    
    f_results = [result for result in onlyfiles if "result" in result]
    f_ids = [result for result in onlyfiles if "ids" in result]
    results =[]
    for result in f_results:
        with open(path+result,"rb") as f:
            results.append(pickle.load(f))

    ids =[]
    for id_ in f_ids:
        with open(path+id_,"rb") as f:
            ids.append(pickle.load(f))
            
    return(results,ids)

def get_id_dict(ids):
    dict_ = {}

    for run in ids:
        for id_ in run:
            try:
                dict_[id_] +=1
            except:
                dict_[id_] = 1

    sorted_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1],reverse=True)}
    return(sorted_)

def get_result_dict(result):
    splits_overall = [[],[],[],[]]
    splits_none = [[],[],[],[]]
    splits_offense = [[],[],[],[]]
    
    for run in result:
        for split in run:
            for i,fold in enumerate(split):
                if i ==0:
                    splits_overall[0].append(fold[0][1])
                    splits_overall[0].append(fold[1][1])
                    splits_overall[0].append(fold[2][1])
                    
                    splits_none[0].append(fold[0][0][0])
                    splits_none[0].append(fold[1][0][0])
                    splits_none[0].append(fold[2][0][0])
                    
                    splits_offense[0].append(fold[0][0][1])
                    splits_offense[0].append(fold[1][0][1])
                    splits_offense[0].append(fold[2][0][1])
                if i ==1:
                    splits_overall[1].append(fold[0][1])
                    splits_overall[1].append(fold[1][1])
                    splits_overall[1].append(fold[2][1])
                    
                    splits_none[1].append(fold[0][0][0])
                    splits_none[1].append(fold[1][0][0])
                    splits_none[1].append(fold[2][0][0])
                    
                    splits_offense[1].append(fold[0][0][1])
                    splits_offense[1].append(fold[1][0][1])
                    splits_offense[1].append(fold[2][0][1])

                if i ==2:
                    splits_overall[2].append(fold[0][1])
                    splits_overall[2].append(fold[1][1])
                    splits_overall[2].append(fold[2][1])
                    
                    splits_none[2].append(fold[0][0][0])
                    splits_none[2].append(fold[1][0][0])
                    splits_none[2].append(fold[2][0][0])
                    
                    splits_offense[2].append(fold[0][0][1])
                    splits_offense[2].append(fold[1][0][1])
                    splits_offense[2].append(fold[2][0][1])
                if i ==3:
                    splits_overall[3].append(fold[0][1])
                    splits_overall[3].append(fold[1][1])
                    splits_overall[3].append(fold[2][1])
                    
                    splits_none[3].append(fold[0][0][0])
                    splits_none[3].append(fold[1][0][0])
                    splits_none[3].append(fold[2][0][0])
                    
                    splits_offense[3].append(fold[0][0][1])
                    splits_offense[3].append(fold[1][0][1])
                    splits_offense[3].append(fold[2][0][1])
                    
    scores = {
        
        "none": splits_none,
        "offense":splits_offense,
        "overall":splits_overall
    }
    return(scores)

def get_statistics(data):
    mean = np.array(list(map(lambda x: statistics.mean(x),data)))
    stdev = np.array(list(map(lambda x: statistics.stdev(x),data)))
    return(mean,stdev)