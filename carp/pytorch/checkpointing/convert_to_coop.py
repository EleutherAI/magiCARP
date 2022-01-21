# import all new model formats here
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders.pool_encoder import *

# import decorator
from carp.pytorch.checkpointing import register_converter, Converter

# import config files
from carp.configs import *

@register_converter("SumTextEncoder", "CARPCoOP")
class ConvertSumTextEncoderV1SumTextEncoderV2(Converter):
    def convert(self, path_orig :str, path_dest : str, **kwargs):
        carp_model = torch.load(path_orig)
        
        coop_model = CARPCoOP(carp_model.config)

        # copy all the huggingface models
        coop_model.passage_encoder = carp_model.encA
        coop_model.review_encoder = carp_model.encB

        # copy the logit scale and projection maps
        coop_model.logit_scale = carp_model.logit_scale
        coop_model.pass_projector = carp_model.projA
        coop_model.rev_projector = carp_model.projB

        # save model
        torch.save(
            coop_model,
            path_dest
        )