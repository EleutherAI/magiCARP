Running Training Script With Poetry
===================================

Assuming a dataset has been installed to the magicarp/carp directory, basic model training can be done by running the
following command from the root magicarp directory:

    poetry run python -m carp.pytorch.training.train \-\-data_path="carp/dataset" \-\-config_path ./configs/base_config.yml  

Use the type flag to specify the type of model architecture you wish to use. For example, to use CARP with `CoOp <https://arxiv.org/abs/2109.01134>`_:

    poetry run python -m carp.pytorch.training.train -\-\data_path="carp/dataset" -\-\config_path ./configs/carp_coop.yml -\-\type carpcoop

Refer to `architectures <notes/architectures>`_ for a comprehensive list of the types of architectures that can be used in training.
