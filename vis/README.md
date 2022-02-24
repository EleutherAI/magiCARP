# CARP Visualizations

**CLOOB CARP**:  
The sample scripts utilize the CLOOB CARP model located on the eye, and expect said checkpoint to be placed in a folder called [CLOOB\_CP](https://mystic.the-eye.eu/public/AI/models/CARP/CLOOB_CARP_Declutr_B.zip). They also assume a dataset of pairs to be used with the text encoders.  

cloobvis.py: Randomly samples from dataset, gets encodings, performs PCA on encodings and scatter plots a 2D visualization. Hovering over points shows corresponding text.

cloobvis.ipynb: Similar to above but interactive and does not have an interactive plot.  

 **TODO**:

- Add visualization of user defined samples
