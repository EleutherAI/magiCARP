# import all old model formats here
from carp.pytorch.legacy.carp_v1 import *

# import all new model formats here
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders.pool_encoder import *

# import decorator
from carp.pytorch.checkpointing import register_converter, Converter

# import config files
from carp.configs import *

@register_converter("SumTextEncoder V1", "SumTextEncoder Momentum V2")
class ConvertSumTextEncoderV1SumTextEncoderV2(Converter):
    def convert(path_orig :str, path_dest : str, **kwargs):
        raise NotImplementedError("TODO: Implement")
        
