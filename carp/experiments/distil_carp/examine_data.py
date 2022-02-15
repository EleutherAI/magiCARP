import pandas as pd
import os

data = []
datapath = 'distil_data/paraphrase_train_crits'
files = os.listdir(datapath)
files.sort()
print(files)
for file in files:
	print(file)
	with open(os.path.join(datapath,file), 'r') as f:
		temp_data = f.readlines()
		temp_data = [ele.split(',') for ele in temp_data]
		data+=temp_data
print(data[8])

datapath = 'distil_data'
story_file = 'train_stories.csv'
story_datapath = os.path.join(datapath, story_file)
with open(story_datapath, 'r') as f:
	data = f.readlines()
print(len(data))
for i, datapoint in enumerate(data):
	if datapoint[0] == ',':
		data[i] = datapoint[1:]

print(data[8])