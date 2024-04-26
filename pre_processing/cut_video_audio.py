from math import ceil
import os
from os import listdir
from os.path import join
import pickle
from statistics import mean
import sys
import numpy as np
import pandas as pd
import torch
from transformers import FlaubertModel, FlaubertTokenizer

CLUSTER="jean-zay"

def getPath(dataset_name):
    set = None
    anno_dir = None
    cut_dictionnary = None

    if dataset_name == "trueness":
        set = ["train", "test"]
        cut_dictionnary={'scene1_sexisme' : [9,176], 'scene2_sexisme' : [18,180],'scene2_confrontation_prise1' : [6,414],'scene2_confrontation2' : [5,420], 'scene2_confrontation3_prise3' : [8,303], 
        'scene2_confrontation4' : [5,433], 'scene2_confrontation5_prise2' : [5,418], 'scene3_sexisme_prise3' : [5,219], 'scene3_confrontation1' : [5,455], 'scene3_confrontation2' : [6,428], 
        'scene3_confrontation3' : [5,433], 'scene3_confrontation5_prise2' : [5,358], 'scene4_racisme_prise2' : [5,231], 'scene5_confrontation1' : [4,327], 'scene6_confrontation2' : [4,436],  
        'scene7_confrontation3' : [6,537], 'scene8_confrontation4' : [5,477],  'scene9_confrontation5' : [5,439]}
    
    else:
        sys.exit("Error in the dataset name")

    if(CLUSTER=="jean-zay"):
        path="/gpfsdswork/projects/rech/urk/uln35en/raw_data/"+dataset_name
        anno_dir = "Annotations/Full/"
        processed_dir = "Annotations/processed/"
    else:
        path = "/storage/raid1/homedirs/alice.delbosc/data/"+dataset_name+"_data/raw_data/"
        anno_dir = "Annotations/Full/"
        processed_dir = "Annotations/processed/"

    return path, processed_dir, anno_dir, set, cut_dictionnary

def cutByIndex(df, begin_cut, end_cut):
    #we cut a the end 
    new_df = df.copy()
    last_index = len(df) - 1
    current_idx = last_index
    while(df.at[current_idx, "begin"] > end_cut):
        new_df = new_df.drop(index = current_idx)
        current_idx = current_idx - 1

    new_df.at[current_idx, "end"] = end_cut

    #we cut the beginning
    first_index = 0
    current_idx = first_index
    while(df.at[current_idx, "end"] < begin_cut):
        new_df = new_df.drop(index = current_idx)
        current_idx = current_idx + 1
    
    new_df.at[current_idx, "begin"] = begin_cut

    return new_df

def shiftValues(df, begin_cut):
    #we shift all the temporal value with the begin value
    df["begin"] = df["begin"] - begin_cut
    df["end"] = df["end"] - begin_cut
    df = df.reset_index(drop = True)
    return df

def cutFile(path, anno_dir, processed_dir):
    print("*"*10, "cutFile", "*"*10)
    cut_dir = join(path,processed_dir, "cut")
    if(not os.path.exists(cut_dir)):
        os.mkdir(cut_dir)
    init_dir = join(path,anno_dir)

    for csv_file in listdir(init_dir):
        key_cut_index = csv_file.find("mic") - 1
        key = csv_file[0:key_cut_index]
    
        df = pd.read_csv(join(init_dir, csv_file),  names=["features", "begin", "end", "speak"])

        df_tokens = df[df['features'] == "Tokens"].reset_index(drop = True)
        #we add the speaking or not speaking features for the audio features
        df_tokens['bool_speak'] = np.where(df_tokens['speak']!= '#', 1, 0)
        df_tokens = df_tokens[["begin", "end", "speak", "bool_speak"]]
        df_tokens.fillna('#', inplace=True)
        df_tokens = df_tokens.replace('nan', '#')

        df_tokensAlign = df[df['features'] == "TokensAlign"].reset_index(drop = True)
        #we add the speaking or not speaking features for the audio features
        df_tokensAlign['bool_speak'] = np.where(df_tokensAlign['speak']!= '#', 1, 0)
        df_tokensAlign = df_tokensAlign[["begin", "end", "speak", "bool_speak"]]
        df_tokensAlign.fillna('#', inplace=True)
        df_tokensAlign = df_tokensAlign.replace('nan', '#')

        if(cut_dictionnary != None):
            begin_cut = cut_dictionnary[key][0]
            end_cut = cut_dictionnary[key][1]

            df_tokens = cutByIndex(df_tokens, begin_cut, end_cut)
            df_tokens = shiftValues(df_tokens, begin_cut)

            df_tokensAlign = cutByIndex(df_tokensAlign, begin_cut, end_cut)
            df_tokensAlign = shiftValues(df_tokensAlign, begin_cut)

            df_tokens.to_csv(join(cut_dir+"/tokens", csv_file), sep=',')
            df_tokensAlign.to_csv(join(cut_dir+"/tokensAlign", csv_file), sep=',')



if __name__ == "__main__":
    dataset_name = sys.argv[1]
    path, processed_dir, anno_dir, set, cut_dictionnary = getPath(dataset_name)

    cutFile(path, anno_dir, processed_dir)