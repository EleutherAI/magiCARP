# import all old model formats here
# import config files
# import decorator
from carp.pytorch.checkpointing import Converter, register_converter

# import all new model formats here


@register_converter("SumTextEncoder V1", "SumTextEncoder Momentum V2")
class ConvertSumTextEncoderV1SumTextEncoderV2(Converter):
    def convert(path_orig: str, path_dest: str, **kwargs):
        raise NotImplementedError("TODO: Implement")
