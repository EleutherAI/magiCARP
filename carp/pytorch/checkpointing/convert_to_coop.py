# import all new model formats here
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders.pool_encoder import *

# import decorator
from carp.pytorch.checkpointing import register_converter, Converter

# import config files
from carp.configs import *

@register_converter("SumTextEncoder", "CARPCoOp")
class ConvertSumTextEncoderV1SumTextEncoderV2(Converter):
    def convert(self, path_orig :str, path_dest : str, **kwargs):
        carp_model = torch.load(path_orig)
        
        CoOp_model = CARPCoOp(carp_model.config)

        # copy all the huggingface models
        CoOp_model.passage_encoder = carp_model.encA
        CoOp_model.review_encoder = carp_model.encB

        # copy the logit scale and projection maps
        CoOp_model.logit_scale = carp_model.logit_scale
        CoOp_model.pass_projector = carp_model.projA
        CoOp_model.rev_projector = carp_model.projB

        # save model
        torch.save(
            CoOp_model,
            path_dest
        )