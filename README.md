# magiCARP: Contrastive Authoring+Reviewing Pretraining

Welcome to the magiCARP API, the test bed used by EleutherAI for performing text/text bi-encoder experiments. 

CARP, or contrastive authorship+reviewing pairings, was first outlined in [Cut the CARP: Fishing for zero-shot story evaluation](https://arxiv.org/abs/2110.03111).

CARP presents a scalable method for performing zero-shot evaluation of stories and other mediums. Current CARP efforts at EleutherAI are primarily focused around controllable code generation. This repository will be updated with more experiments over the coming months as we try varying CARP architectures.


To train a model, run 
```poetry run python -m carp.pytorch.train --data_path="carp/dataset" --config_path ./base_config.yml```

Finetuning via CoOp now available. Preference learning coming soon!