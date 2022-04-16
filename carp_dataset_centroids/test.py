import torch
import pandas as pd

def test_1():
	pos = torch.load('dataset_centroid_dists_pos.pt')
	print(pos.shape)
	pos = pos.type(torch.float32)
	print(pos[0])

	softmaxed_positives = torch.softmax(pos, dim=1)
	print(softmaxed_positives.shape)
	print(torch.sum(softmaxed_positives[0]))
	print(softmaxed_positives[0])

	argmaxes = torch.max(softmaxed_positives, dim=-1).values
	res = argmaxes >= .01
	filtered_softmax = softmaxed_positives[res]
	print(filtered_softmax.shape)

	passages = pd.read_csv('dataset.csv')['passages']


	metalabels_pd = pd.read_csv('passage_metalabel_dataset.csv')
	passages = list(metalabels_pd["passages"])
	# get the target distributions
	reviews = torch.tensor(metalabels_pd[[str(i) for i in range(92)]].values.tolist())
	print(reviews.shape)
	print(reviews[0])
	filtered_reviews = reviews[torch.max(reviews, dim=-1).values > .5]
	print(filtered_reviews.shape)

def test_2():
	pos = torch.load('new_positive_dists.pt')
	pos = pos.type(torch.float32)
	print(pos.shape)
	print(pos[0])
	print(pos[torch.abs(pos) > 1].shape)

	softmaxed_new_positives = torch.softmax(pos, dim=-1)
	torch.save(softmaxed_new_positives, 'softmaxed_new_positives.pt')
	print(softmaxed_new_positives.shape)
	print(softmaxed_new_positives[0])
	print(softmaxed_new_positives[softmaxed_new_positives > .01].shape)

def test_3():
	dataset = torch.load('cosine_sim/metalabel_data/dataset_centroid_dists_overall.pt')
	dataset = dataset.type(torch.float32)
	print(dataset.shape)
	print(dataset[0])

	softmaxed_cosine_sim = torch.softmax(dataset, dim=-1)
	print(softmaxed_cosine_sim[0])
	print(softmaxed_cosine_sim[softmaxed_cosine_sim > .01].shape)
	torch.save(softmaxed_cosine_sim, 'softmaxed_cosine_sim.pt')

def test_4():
	csv = pd.read_csv('alignment.csv')
	passages = csv['story'].values.tolist()
	reviews = csv.loc[:, csv.columns != 'story'].values.tolist()
	reviews = torch.tensor(reviews)
	softmax = torch.softmax(reviews, dim=-1)
	print(reviews[4])
	print(passages[4])
	print(len(passages), len(reviews))
	print(softmax[0])

def test_5():
	labels = ['chapter', 'scene', 'description', 'character names', 'paragraph', 'dialogue', 'punctuation', 'lines', 'sentences', 'pov', 'character asking question', 'story', 'conversation', 'cut this', 'backstory', 'single character', 'emotional response to story event', 'openings and endings', 'use of action', 'character age', 'humor']
	print(len(labels))
	csv = pd.read_csv('filtered_semantic_metadataset/dataset.csv')
	passages = csv['passages']
	print(len(passages))
	data = torch.load('filtered_semantic_metadataset/dataset_centroid_dists_overall.pt')
	data = data.type(torch.float32)
	print(data.shape)
	print(data[0])

	softmaxed = torch.softmax(data, dim=-1)
	print(softmaxed[0])
	#torch.save(softmaxed, 'softmaxed_filtered_semantic.pt')

	maxes = torch.argmax(softmaxed, dim=-1)
	classes = {index:[] for index in range(22)}
	for m, p in zip(maxes, passages):
		classes[m.item()].append(p)
	with open('labels.txt','w') as f:
		for i, bucket in classes.items():
			f.write(f'Section {i}\n\n')
			for story in bucket:
				f.write(story+'\n')

def test_6():
	csv = pd.read_csv('og_metalabel_dataset/passage_metalabel_dataset.csv')
	print(csv.head())

def test_7():
	csv = pd.read_csv('fixed_semantic_filtered/dataset.csv')
	passages = csv['passages']
	data = torch.load('fixed_semantic_filtered/dataset_centroid_dists_overall.pt')
	print(len(passages))
	print(data.shape)


if __name__ == "__main__":
	test_7()