import csv
from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *

def read_dataset_component(filepath):
	data = list()
	with open(filepath, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
		for row in reader:
			data.append(row[1])
	return data

config = CARPConfig.load_yaml('configs/carp_cloob.yml')
carp_model = CARPCloob(config.model)
carp_model.load('ckpts/CLOOB CARP Declutr B/')

