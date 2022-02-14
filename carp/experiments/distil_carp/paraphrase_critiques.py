from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import csv
import sys
from tqdm import tqdm

tokenizer_pegasus = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
model_pegasus = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase').half().to("cuda")

def read_dataset_component(filepath):
		data = list()
		with open(filepath, newline='') as csvfile:
				reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
				for row in reader:
						data.append(row[1])
		return data

num_beams = 5
def get_review_ensemble(input_text):
	batch = tokenizer_pegasus(input_text,truncation=True,padding='longest', max_length=60, return_tensors="pt").to("cuda")
	translated = model_pegasus.generate(**batch,max_length=60, num_beams=num_beams, num_return_sequences=num_beams, temperature=1.5)
	return tokenizer_pegasus.batch_decode(translated, skip_special_tokens=True)

def write_dataset_csv(data, filepath):
    with open(filepath, mode='w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(data)

filepath = '/mnt/raid/users/AlexH/magiCARP/carp/pytorch/data/utils/train_crits.csv'
data = read_dataset_component(filepath)
batch_size = 100
num_batches = (len(data) + batch_size - 1) // batch_size
output_file = 'paraphrase_train_crits.csv'
write_thresh = 10000
temp_csv = []
print(len(data))
for i in tqdm(range(num_batches)):
	cur_batch_size = min(batch_size, len(data)-batch_size*i)
	batch = data[i*batch_size:i*batch_size+cur_batch_size]
	#print(batch)
	num_paraphrases = 5
	paraphrases = get_review_ensemble(batch)
	reshaped_paraphrases = []
	for j in range(len(paraphrases)//num_paraphrases):
		reshaped_paraphrases.append([])
		for k in range(num_paraphrases):
			reshaped_paraphrases[-1].append(paraphrases[j*num_paraphrases+k])
	#print(reshaped_paraphrases)
	temp_csv += reshaped_paraphrases
	if (i+1) % write_thresh == 0:
		print("WRITING TO CSV")
		cur_output_file = output_file+f"_{i}"
		write_dataset_csv(temp_csv ,cur_output_file)
		temp_csv = []
write_dataset_csv(temp_csv, output_file)

