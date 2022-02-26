from carp.pytorch.encoder import SumTextEncoder, TextEncoders

model_path = "haisongzhang/roberta-tiny-cased"
model_arch = "roberta"


def test_text_encoders():
    text_encoders = TextEncoders()
    assert text_encoders.sum == text_encoders["sum"]
    encoder = text_encoders["sum"](model_path, model_arch)
    assert isinstance(encoder, SumTextEncoder)
