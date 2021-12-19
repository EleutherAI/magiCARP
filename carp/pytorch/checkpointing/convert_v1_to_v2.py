# import all old model formats here
from carp.pytorch.legacy.carp_v1 import *

# import all new model formats here
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders.encoder import *

# import decorator
from carp.pytorch.checkpointing import register_converter, Converter

# import config files
from carp.configs import *

from carp.pytorch.model.encoders import get_encoder, get_encoder_names

encoder_names = get_encoder_names()

for enc_name in encoder_names:
    @register_converter("SumTextEncoder V1", enc_name + " V2")
    class ConvertSumTextEncoderV1SumTextEncoderV2(Converter):
        def convert(self, path_orig :str, path_dest : str, **kwargs):
            v1_model = ContrastiveModelV1(TextEncoderV1(), TextEncoderV1())
            v1_model.load_state_dict(torch.load(path_orig))

            # initialize the model on cpu using roughly the same parameters as the v1 model
            v2_config = ModelConfig(latent_dim=2048,
                proj_dropout=0.1,
                linear_projection=True,
                model_path="roberta-large",
                model_arch="roberta",
                encoder_type=enc_name,
                momentum=0.,
                device="cpu",
                grad_clip=1.0)

            v2_model = CARP(v2_config)

            # copy all the huggingface models
            v2_model.passage_encoder = v1_model.encA
            v2_model.review_encoder = v1_model.encB

            # copy the logit scale and projection maps
            v2_model.logit_scale = v1_model.logit_scale
            v2_model.pass_projector = v1_model.projA
            v2_model.rev_projector = v1_model.projB

            # save model
            torch.save(
                v2_model.state_dict(),
                path_dest
            )