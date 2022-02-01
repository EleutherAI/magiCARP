Preparing The Dataset
=========================
magiCARP contains multiple data pipelines which can be specified through the 
config file (see `configuring carp <config>`_).
The base CARP architecture is trained using data structrued in the form of
tuples containing some passage and a corresponding review.


Vanilla CARP 
============
.. autoclass:: carp.pytorch.data.utils.data_util.BatchElement
    :members:

.. autoclass:: carp.pytorch.data.BaseDataPipeline
    :members:

CARP MLM
============

.. autoclass:: carp.pytorch.data.mlm_pipeline.MLMBatchElement
    :members:

.. autoclass:: carp.pytorch.data.MLMDataPipeline
    :members:

CARP COOP 
===========

.. autoclass:: carp.pytorch.data.scarecrow_pipeline.ScarecrowTargetElement
    :members:

.. autoclass:: carp.pytorch.data.ScarecrowDataPipeline
    :members:

Utility
==========

.. autofunction:: carp.pytorch.data.utils.data_util.create_tok
.. autofunction:: carp.pytorch.data.utils.data_util.chunkBatchElement

