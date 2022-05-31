import torch
import pandas as pd
import torch.nn.functional as F

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

def test_8():
	csv = pd.read_csv('fixed_semantic_filtered/dataset.csv')
	passages = csv['passages']
	data = torch.load('fixed_semantic_filtered/dataset_centroid_dists_overall.pt')
	data = data.type(torch.float32)
	argmaxes = torch.argmax(data, dim=-1)
	one_hots = F.one_hot(argmaxes, num_classes=21).type(torch.float32)
	print(one_hots.shape)
	print(argmaxes[0])
	print(one_hots[0])

	torch.save(one_hots, 'thresholded.pt')

def test_9():
	csv = pd.read_csv('metalabel_data/dataset.csv')
	passages = csv['passages']
	data = torch.load('metalabel_data/dataset_centroid_dists_overall.pt').type(torch.float32)
	print(csv.head())
	print(data.shape)
	labels = ['chapter', 'scene', 'description', 'character names', 'paragraph', 'dialogue', 'punctuation', 'lines', 'sentences', 'pov', 'character asking question', 'story', 'conversation', 'cut this', 'backstory', 'single character', 'emotional response to story event', 'openings and endings', 'use of action', 'character age', 'humor']
	print(passages[0])
	print(data[0])
	#how does carp actually rate this data?
	target_critiques = ['scene', 'description', 'character asking question', 'conversation', 'emotional response to story event', 'openings and endings', 'use of action', 'humor']
	target_indices = []
	for crit in target_critiques:
		target_indices.append(labels.index(crit))
	target_indices.sort()
	print(target_indices)
	#suggested: ['chapter', 'scene', 'description', 'character names', 'paragraph', 'lines', 'character asking question', 'conversation', 'single character', 'openings and endings']
	#I really want representative text of the critique. No noise
	filtered_cols = data[:,target_indices]
	print(filtered_cols.shape)
	curated_data = []
	curated_passages = []
	COSINE_THRESH = .3
	for i, cosines in enumerate(filtered_cols):
		print(i)
		relevant_inds = cosines > COSINE_THRESH #Is more than .3 cosine similarity(average of alignments)
		if torch.sum(relevant_inds) > 0:
			relevant_inds = relevant_inds.type(torch.float32)
			threshed_cosines = torch.where(relevant_inds > 0, cosines, torch.tensor(float('-inf')).type(torch.float32))
			softmaxed_threshed_cosines = torch.softmax(threshed_cosines,dim=0)
			curated_data.append(softmaxed_threshed_cosines)
			curated_passages.append(passages[i])

	curated_data = torch.stack(curated_data)
	print(len(curated_passages))
	print(curated_data.shape)

	df = pd.DataFrame({'passages':curated_passages})
	df.to_csv('curated_passages.csv')
	torch.save(curated_data, 'curated_data.pt')

def test_10():
	csv = pd.read_csv('alignment_data/alignment.csv')
	print(csv.head())
	print(len(csv))
	logits = torch.tensor(csv.loc[:, csv.columns != 'story'].values.tolist())
	dist = torch.softmax(logits,dim=1)
	torch.save(dist, 'alignment_dist.pt')
	print(dist[:5])

def clean_carp_l_metadataset():
	csv = pd.read_csv('carp_l_metadataset/dataset.csv')
	passages = csv['passages']
	#passages = passages[:10000]
	dataset = torch.load('carp_l_metadataset/dataset_centroid_dists_overall.pt')
	#dataset = dataset[:10000]
	print(len(csv))
	print(dataset.shape)
	print(csv.head())
	print(dataset[0])
	labels = []
	with open('carp_l_metadataset/user_captions.txt') as f:
		lines = f.readlines()
		for line in lines:
			stripped_line = line.strip('\n')
			if stripped_line != '?':
				labels.append(stripped_line)
	print(labels)
	useful_labels = ['characters laughing or finding things funny', 'imagery/descriptions', 'characters asking questions', 'praying/religion/church', 'accident/bad scenarios', 'character\'s internal monologues and thoughts', 'crimes/vilations', 'fighting', 'music', 'family']
	useful_label_indices = [labels.index(useful_label) for useful_label in useful_labels]
	print(useful_label_indices)
	useful_dataset = dataset[:, useful_label_indices]
	print(useful_dataset.shape)

	curated_data = []
	curated_passages = []
	COSINE_THRESH = .5
	CLASS_SIZE_TARGET = 1000
	label_counts = torch.zeros(len(useful_labels))
	for i, cosines in enumerate(useful_dataset):
		if torch.sum(label_counts) == CLASS_SIZE_TARGET*label_counts.shape[0]:
			break
		print(i)
		relevant_inds = cosines > COSINE_THRESH #Is more than .3 cosine similarity(average of alignments). Filter out totally irrelevant passages
		if torch.sum(relevant_inds) > 0:
			#We are only representing a point masses to single clusters
			max_ind = torch.argmax(cosines)
			if label_counts[max_ind] < CLASS_SIZE_TARGET:
				new_dist = torch.zeros_like(cosines).type(torch.float32)
				new_dist[max_ind] = 1.0
				curated_data.append(new_dist)
				curated_passages.append(passages[i])
				label_counts[max_ind] = label_counts[max_ind] + 1

	curated_data = torch.stack(curated_data)
	print(len(curated_passages))
	print(curated_data.shape)
	print(curated_data[4])
	print(curated_passages[4])

	df = pd.DataFrame({'passages':curated_passages})
	df.to_csv('curated_passages.csv')
	torch.save(curated_data, 'curated_data.pt')
	label_counts = torch.zeros(len(useful_labels))
	print("")
	print("Looking at sample passages from cluster")
	for datapoint, passage in zip(curated_data, curated_passages):
		label = torch.argmax(datapoint)
		label_counts[label] += 1
	print(label_counts)

def test_carp_l_metadataset():
	passages = pd.read_csv('carp_l_metadataset/curated_passages.csv')
	reviews = torch.load('carp_l_metadataset/curated_data.pt')
	print(len(passages))
	print(reviews.shape)
	print(passages['passages'][0])
	print(reviews[0])

def make_alignment_dataset():
	data = pd.read_csv('alignment_data/prompt1_new_alighment_test.csv')
	print(len(data))
	print(data.head())
	passages = data['story']
	passages.to_csv('alignment_passages.csv')
	dist = torch.tensor(data.loc[:, data.columns != 'story'].values.tolist())
	print(dist.shape)
	dist = torch.softmax(dist, dim=1)
	print(dist[0])
	print(passages[0])
	counts = torch.zeros(3)
	for ele in dist:
		max_ind = torch.argmax(ele)
		counts[max_ind] += 1
	print(counts)
	torch.save(dist, 'curated_data.pt')

def examine_coop_dataset():
	dists = torch.load('carp_l_metadataset/curated_data.pt')
	stories = pd.read_csv('carp_l_metadataset/curated_passages.csv')['passages']
	useful_labels = ['characters laughing or finding things funny', 'imagery/descriptions', 'characters asking questions', 'praying/religion/church', 'accident/bad scenarios', 'character\'s internal monologues and thoughts', 'crimes/vilations', 'fighting', 'music', 'family']
	label_to_index = {label:i for i, label in enumerate(useful_labels)}
	for i in range(len(stories)):
		datapoint = dists[i]
		if(datapoint[label_to_index['characters laughing or finding things funny']] > .5):
			print(datapoint)
			print(stories[i])



if __name__ == "__main__":
	#clean_carp_l_metadataset()
	#test_carp_l_metadataset()
	examine_coop_dataset()