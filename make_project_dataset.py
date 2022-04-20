import csv
from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
import torch
from tqdm import tqdm
import pandas as pd

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

i = 0
while i < len(passages):
	passage_encs = None
	crit_encs = None
	stored_passages = []
	stored_crits = []
	print(i)
	passage_chunk = passages[i:i+CHUNK_SIZE]
	crit_chunk = crits[i:i+CHUNK_SIZE]
	i += CHUNK_SIZE
	for j in tqdm(range(CHUNK_SIZE // BATCH_SIZE)):
		try:
			passage_batch = passage_chunk[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
			passage_batch_encs = encode_passages(passage_batch)
			crit_batch = crit_chunk[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
			crit_batch_encs = encode_reviews(crit_batch)

			if passage_encs == None:
				passage_encs = passage_batch_encs
			else:
				passage_encs = torch.cat([passage_encs, passage_batch_encs])

			if crit_encs == None:
				crit_encs = crit_batch_encs
			else:
				crit_encs = torch.cat([crit_encs, crit_batch_encs])

			stored_passages += passage_batch
			stored_crits += crit_batch
		except:
			print("Caught runtime error")
	torch.save(passage_encs, f'encoded_dataset/passages/passage_encodings_{i}.pt')
	torch.save(crit_encs, f'encoded_dataset/crits/crit_encodings_{i}.pt')
	text_dict = {'passages':stored_passages, 'crits':stored_crits}
	df = pd.DataFrame(text_dict)
	df.to_csv(f'encoded_dataset/text/text_{i}.csv', index=False)

	exit()


