from typing import Iterable

from transformers import AutoFeatureExtractor, AutoConfig, AutoModel
import PIL

from carp.pytorch.model.encoders import (
    BaseEncoder,
    BaseEncoderOutput,
    register_encoder,
)

@register_encoder
class ImageEncoder(BaseEncoder):
    def __init__(self, model_path : str, model_arch : str, tokenizer_path : str = None):
        super().__init__(model_path, model_arch, tokenizer_path, skip_init = True)

        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoFeatureExtractor.from_pretrained(tokenizer_path)

    def call_tokenizer(self, batch : Iterable[PIL.Image]):
        return self.tokenizer(
            batch,
            return_tensors = "pt"
        )
