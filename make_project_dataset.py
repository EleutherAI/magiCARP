import csv
from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
import torch
from tqdm import tqdm

def read_dataset_component(filepath):
	data = list()
	with open(filepath, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
		for row in reader:
			data.append(row[1])
	return data

passages = read_dataset_component('carp_data/train_stories.csv')
crits = read_dataset_component('carp_data/train_crits.csv')

config = CARPConfig.load_yaml('configs/carp_cloob.yml')
carp_model = CARPCloob(config.model)
carp_model.load('ckpts/CLOOB CARP Declutr B/')
carp_model.to('cuda')

tokenizer = carp_model.passage_encoder.call_tokenizer

def encode_passages(txt_batch):
	tok_out = tokenizer(txt_batch)
	x = tok_out["input_ids"]
	mask = tok_out["attention_mask"]
	enc_input = BatchElement(x, mask)

	with torch.no_grad():
		encs = carp_model.encode_passages(enc_input).hidden
	encs = encs.cpu().half()
	return encs

def encode_reviews(txt_batch):
	tok_out = tokenizer(txt_batch)
	x = tok_out["input_ids"]
	mask = tok_out["attention_mask"]
	enc_input = BatchElement(x, mask)

	with torch.no_grad():
		encs = carp_model.encode_reviews(enc_input).hidden
	encs = encs.cpu().half()
	return encs

CHUNK_SIZE = 100000
BATCH_SIZE = 500

i = 100000
encs = None
while i < len(passages):
	print(i)
	passage_chunk = passages[i:i+CHUNK_SIZE]
	i += CHUNK_SIZE
	for j in tqdm(range(CHUNK_SIZE // BATCH_SIZE)):
		try:
			batch = passage_chunk[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
			batch_encs = encode_passages(batch)
			if encs == None:
				encs = batch_encs
			else:
				encs = torch.cat([encs, batch_encs])
		except:
			print("Caught runtime error")
	torch.save(encs, f'encoded_dataset/passages/passage_encodings_{i}')
	del passage_chunk

i = 0
encs = None
while i < len(crits):
	print(i)
	crit_chunk = crits[i:i+CHUNK_SIZE]
	i += CHUNK_SIZE
	for j in tqdm(range(CHUNK_SIZE // BATCH_SIZE)):
		try:
			batch = crit_chunk[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
			batch_encs = encode_reviews(batch)
			if encs == None:
				encs = batch_encs
			else:
				encs = torch.cat([encs, batch_encs])
		except:
			print("Caught runtime error")
	torch.save(encs, f'encoded_dataset/crits/crit_encodings_{i}')
	del crit_chunk

