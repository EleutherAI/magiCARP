from typing import Iterable
from torchtyping import TensorType

from transformers import AutoFeatureExtractor, AutoConfig, AutoModel
import PIL
import torch.nn.functional as F

from carp.pytorch.model.encoders import (
    BaseEncoder,
    BaseEncoderOutput,
    register_encoder,
)

@register_encoder
class VITImageEncoder(BaseEncoder):
    def __init__(self, model_path : str, model_arch : str, tokenizer_path : str = None):
        super().__init__(model_path, model_arch, tokenizer_path, skip_init = True)

        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoFeatureExtractor.from_pretrained(tokenizer_path)

    def call_tokenizer(self, batch : Iterable[PIL.Image.Image]):
        return self.tokenizer(
            batch,
            return_tensors = "pt"
        ).pixel_values
    
    def forward(
        self,
        pixel_values,
        tokenize : bool = False, mask_sum : bool = True,
        inputs_embeds : bool = False,
    ) -> TensorType["batch", "embed_dim"]:
        if tokenize:
            pixel_values = self.call_tokenizer(pixel_values)
        out = self.model(output_hidden_states = True, return_dict = True, pixel_values = pixel_values)

        hidden : TensorType["batch", "n_patches+1", "embed_dim"] = out["last_hidden_state"]

        return BaseEncoderOutput(F.normalize(hidden.sum(1))) # Sum along the patches
