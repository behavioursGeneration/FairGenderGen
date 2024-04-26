import argparse
from os.path import join, isfile
from os import listdir
import sys
import time
import pandas as pd
import numpy as np
import pickle
import librosa
from transformers import Wav2Vec2Processor
import torch
from transformers import HubertModel


CLUSTER="jean-zay"

    
def complete_set(init_output_path, output_path):    
    for file in listdir(output_path):
        with open(join(output_path, file), 'rb') as f:
            final_dict = pickle.load(f)
        key = file[0:-2]
        print("process of", key)
        #complete here

        with open(join(output_path, file), 'wb') as f:
            pickle.dump(final_dict, f)
        del final_dict


def getPath(dataset_name, moveToZero, segment_length):
    if(CLUSTER=="jean-zay"):
        general_path = "/gpfsdswork/projects/rech/urk/uln35en/"
        dataset_path = general_path + "raw_data/"+dataset_name+"/"
        init_output_path = dataset_path + "/final_data/"

        if(moveToZero):
            output_path = join(init_output_path, str(segment_length), "moveSpeakerOnly")
        else:
            output_path = join(init_output_path, str(segment_length), "none")
        
        audio_path = dataset_path + "audio/full/"
        visual_path = dataset_path + "video/processed/" 
        ipu_path = dataset_path + "annotation/processed/ipu_with_tag/"
        prosody_path = dataset_path + "audio/processed/"
        details_file = dataset_path + "details.xlsx"
        details_df = pd.read_excel(details_file)
        data_details = details_df.set_index("nom").to_dict(orient='index')
    else:
        sys.exit("Error in the cluster name")
    return init_output_path, output_path, audio_path, visual_path, ipu_path, prosody_path, data_details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset')
    parser.add_argument('-zeroMove', action='store_true')
    parser.add_argument('-segment', type=int, default=2)
    args = parser.parse_args()

    dataset_name = args.dataset
    moveToZero = args.zeroMove
    segment_length = args.segment #secondes
    

    overlap_train = round(0.1 * segment_length,2) 
    overlap_test = 0
    print("length for train", segment_length, "overlap for train:", overlap_train, "overlap for test", overlap_test)

    init_output_path, output_path, audio_path, visual_path, ipu_path, prosody_path, data_details = getPath(dataset_name, moveToZero, segment_length)
    complete_set(init_output_path, output_path)
    print("*"*10, "end of creation", "*"*10)

    return 0

if __name__ == "__main__":
    sys.exit(main())
