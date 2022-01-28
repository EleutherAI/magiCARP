Preparing The Dataset
=========================
magiCARP contains multiple data pipelines which can be specified through the config file (see `configuring carp <config>`_). The base CARP architecture is trained using data structrued in the form of tuples containing some passage and a corresponding review.
More information on what the pipelines should be used for is provided in architectures.

**Data Pipelines** 

**Base Data Pipeline**

.. autoclass:: carp.pytorch.data.register_datapipeline
    :members:

**MLM Data Pipeline**

**Scarecrow Data Pipeline**