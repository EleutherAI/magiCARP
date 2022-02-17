import pandas as pd
import os
import csv

def read_dataset_component(filepath):
    data = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            data.append(row[1])
    return data

def read_paraphrase_component(filepath):
    data = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            data.append(row)
    return data

path = 'distil_data'

crit_data = []
crit_datapath = path+'/paraphrase_train_crits'
files = os.listdir(crit_datapath)
files.sort()
for file in files:
	print(file)
	filepath = os.path.join(crit_datapath, file)
	crit_data_chunk = read_paraphrase_component(filepath)
	crit_data += crit_data_chunk

story_file = 'train_stories.csv'
story_datapath = os.path.join(path, story_file)
story_data = read_dataset_component(story_datapath)

orig_crit_file = 'train_crits.csv'
orig_crit_datapath = os.path.join(path, orig_crit_file)
orig_crit_data = read_dataset_component(orig_crit_datapath)

print("NUM STORIES: ", len(story_data))
print("NUM CRITIQUE LISTS: ", len(crit_data))
print("NUM ORIG CRITS: ", len(orig_crit_data))
print("NUM CRITIQUES PER: ", len(crit_data[1]))
print(story_data[1])
print(crit_data[1])