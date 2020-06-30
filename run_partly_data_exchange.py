import pandas as pd
import numpy as np
import pickle
import argparse

from helper.data_loading import *
from helper.preprocessing import *
from helper.cross_val_model import *
from helper.classifier_helper import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output", help="output")
    parser.add_argument("-d", "--dataset", dest="dataset",help="dataset")
    
    parser.add_argument("-f", "--folds_per_run", dest="folds_per_run", help="folds_per_run", default=5)
    parser.add_argument("-r", "--runs", dest="runs",help="runs", default=1)
    parser.add_argument("-s", "--steps_data_exchange", dest="steps_data_exchange",help="steps_data_exchange", default=5)

    args = parser.parse_args()
    
    print("Args:\n",args)
    
    if(args.dataset=="LEFT"):
        dataset = pd.read_pickle('data/LEFT_with_topic_distribution.pkl')
    elif(args.dataset=="RIGHT"):
        dataset = pd.read_pickle('data/RIGHT_with_topic_distribution.pkl')
    elif(args.dataset=="NEUTRAL"):
        dataset = pd.read_pickle('data/NEUTRAL_with_topic_distribution.pkl')
    
    germEval = pd.read_pickle('data/GERMEVAL_with_topic_distribution.pkl')
    germEval = germEval.rename(columns={'label_1': 'label'})

    germEval_topic_distribution = get_topic_distribution_over_dataset(germEval.loc[germEval.label=="OTHER",:])
    print("germEval topic distribution:\n",germEval_topic_distribution, "\n")

    data_pool = create_sample_pool(dataset, germEval_topic_distribution, sample_factor = 5)

    print("\nStart dataset:\n")
    result, ids_ = run_experiment_partly_data_exchange(data_pool,
                                                       germEval,
                                                       germEval_topic_distribution,
                                                       folds_per_run=int(args.folds_per_run),
                                                       runs=int(args.runs),
                                                       steps_data_exchange=int(args.steps_data_exchange)) 
    
    
    with open(args.output+"_result.pkl", 'wb') as f:
        pickle.dump(result, f)

    with open(args.output+"_ids.pkl", 'wb') as f:
        pickle.dump(ids_, f)
