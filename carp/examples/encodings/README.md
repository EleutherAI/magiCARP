# Usage

The scripts in this folder generate encodings en-masse from the Story-Critique dataset mentioned in the CARP paper. By default, the scripts use the CARP-L model (with Roberta-Large encoders). They expect the model checkpoint to be in the path "checkpoints/carp_L". The checkpoint for CARP-L trained on the Story-Critique dataset can be found on [the eye](https://the-eye.eu/public/AI/models/CARP/CARP_L.pt). Note that the scripts have relatively long runtimes, so it is reccomended to encode a moderately sized sample of the dataset. These encodings can then be used for visualization or clustering. Additionally, indices for which dataset items were encoded are saved along with the encodings. 

Note that both scripts use checkpoints in case of crashes or interruptions. In order to generate new encodings, please specify FRESH or fresh as an argument.   
i.e.  
python -m carp.examples.encodings.encode_reviews FRESH
  
Also, keep in mind that because of how saving works, the generated encoding tensor may contain many rows of zero vectors at its end. Code using the encodings must deal with these accordingly. It is also in float16, so a casting to float32 is reccomended when doing any computations downstream with the generated encodings.
