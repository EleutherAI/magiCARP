python train.py --type carpcoop

python -m carp.pytorch.training.train \
       --data_path="carp/dataset/passage_metalabel_dataset.csv" \
       --config_path ./configs/large_coop.yml \
	--type carpcoop