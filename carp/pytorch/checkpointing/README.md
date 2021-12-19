Welcome to the conversion tool, your one stop shop for converting old CARP checkpoints into new CARP checkpoints. To execute a conversion, run the following script

```python -m carp.pytorch.checkpointing.convert --origin_path [ORIGIN PATH] --destination_path [DESTINATION_PATH] --origin_type [ORIGIN TYPE] --destination_type [DESTINATION TYPE]```

Where origin_path, destination_path refers to the directory of the origin checkpoint and where you would like to save the converted checkpoint to respectively. Origin type refers to the kind of checkpoint you are converting from, where as destination type refers to the type of checkpoint you are converting to. 

To create new converters, register them using the ```register_converter``` decorator.

Only important thing is to copy over the text encoders and the projection matrices currently but, future models might be significantly more complicated so it might be useful to have use case specific conversion files.

The current conversion tool requires you to store both models in RAM concurrently. At current model sizes this is fine, and it should be ok for [REDACTED] which has [REDACTED]GB of RAM, but we probably need a better solution in the long run.
