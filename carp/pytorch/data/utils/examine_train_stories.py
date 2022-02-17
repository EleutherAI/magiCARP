import pandas as pd
import csv

with open('train_stories.csv', 'r') as f:
	data = f.readlines()

print(len(data))
print(data[:5])
data = pd.read_csv('train_stories.csv')
print(data.shape[0])


def read_dataset_component(filepath):
		data = list()
		with open(filepath, newline='') as csvfile:
				reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
				for row in reader:
						data.append(row[1])
		return data

data = read_dataset_component('train_stories.csv')
print(len(data))