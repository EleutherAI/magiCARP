from typing import List, Callable, Iterable, Tuple, Any

from torch import nn, no_grad
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from catalyst.data import DistributedSamplerWrapper

from carp.configs import ModelConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.scalability_utils import print_rank_0
from carp.pytorch.training.trainer import BaseTrainer, register_trainer
from carp.util import generate_indices

from carp.pytorch.model.encoders import get_encoder

from carp.pytorch.data import BaseDataPipeline, get_datapipeline
from carp.pytorch.data.utils.data_util import GenericBatchElement, split_batch_element, batch_elem_to_dict

patch_typeguard()

@typechecked
@register_architecture
class BaseAgnosticModel(BaseModel):
    """
    Mode-agnostic architecture for any bi-encoder contrastive model. Mostly identical to CARP

    :param config: Unlike other configs, expects two arguments for fields like config.encoder_type,
    formatted as encoder_a|encoder_b (i.e. for encoder_type, model_path, model_arch, tokenizer_path).
    tokenizer_path for non text should be path to a valid feature extractor for the expected modality
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config, skip_init = True)

        enc_type = self.config.encoder_type.split("|")
        model_path = self.config.model_path.split("|")
        model_arch = self.config.model_arch.split("|")
        tokenizer_path = self.config.model_arch.split("|")

        pass_enc_class = get_encoder(enc_type[0])
        rev_enc_class = get_encoder(enc_type[1])

        self.passage_encoder = pass_enc_class(
            model_path[0], model_arch[0], tokenizer_path[0]
        )

        self.review_encoder = rev_enc_class(
            model_path[1], model_arch[1], tokenizer_path[1]
        )

        self.latent_dim = self.config.latent_dim

        self.pass_projector, self.rev_projector = self._make_projection_layers(
            self.config
        )

        self.logit_scale = nn.Parameter(
                torch.ones([], device=self.config.device)
                * torch.log(torch.tensor([1 / 0.07], device=self.config.device))
        )
        self.logit_scale.requires_grad = False
        self.clamp_min = torch.log(
            torch.tensor([1 / 100], device=self.config.device)
        )
        self.clamp_max = torch.log(torch.tensor([100], device=self.config.device))
    
    def forward(
        self,
        passages: GenericBatchElement,
        reviews: GenericBatchElement,
        config: TrainConfig,
    ):
        """
        Forward pass of the model.
        :param passages: The passages to encode.
        :param reviews: The reviews to encode.
        :param config: The training configuration.
        :return: The encodings of the passages and reviews w/o gradients.
        """
        microbatch_inds = generate_indices(
            passages.input_ids.shape[0], config.microbatch_size, shuffle=False
        )

        # Into microbatches


        pass_mbs: List[GenericBatchElement] = split_batch_element(passages, microbatch_inds)
        rev_mbs: List[GenericBatchElement] = split_batch_element(reviews, microbatch_inds)

        # Initially get all encodings without grad
        pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)

        # compute accuracy
        forward_acc = self.compute_accuracy(torch.cat(pass_encs), torch.cat(rev_encs))
        top_k_acc = self.compute_top_k_accuracy(
            torch.cat(pass_encs), torch.cat(rev_encs)
        )

        return {
            "pass_mbs": pass_mbs,
            "pass_encs": pass_encs,
            "rev_mbs": rev_mbs,
            "rev_encs": rev_encs,
            "forward_acc": forward_acc,
            "top_k_acc": top_k_acc,
        }

    def _embed_data(
        self,
        x : GenericBatchElement,
        encoder,
        projector,
        normalize = False
    ):
        """
        Embed data with given encoder and projector
        :return: Embedded data
        """
        x = encoder(**batch_elem_to_dict(x))
        x.hidden = projector(x.hidden)

        if normalize:
            x.hidden = F.normalize(x.hidden)
        return x
    
    def calculate_embeddings(
        self,
        passages: Iterable[Tuple[BatchElement]],
        reviews: Iterable[Tuple[BatchElement]],
        return_only_embeddings: bool = True,
    ):
        # Get encodings without grad
        with no_grad():
            pass_encs = [self.encode_passages(p) for p in passages]
            rev_encs = [self.encode_reviews(r) for r in reviews]

        # if we only need the embeddings, fetch them
        if return_only_embeddings:
            pass_encs = list(map(lambda x: x.hidden, pass_encs))
            rev_encs = list(map(lambda x: x.hidden, rev_encs))
        return pass_encs, rev_encs

@register_trainer
class BaseAgnosticTrainer(BaseTrainer):
    def train_deepspeed_step(
        self,
        passages: GenericBatchElement,
        reviews: GenericBatchElement,
        config: TrainConfig,
    ):
        """
        Train the model on a single batch using deepspeed.
        :param passages: The passages to encode.
        :param reviews: The reviews to encode.
        :param config: The training configuration.
        :return: The loss and accuracy of the model on the batch.
        """
        with self.autocast():
            forward_output = self.model(passages, reviews, config)

        rev_encs, all_rev_encs, rev_offset = self.contrastive_parallel_all_gather(
            forward_output["rev_encs"]
        )
        pass_encs, all_pass_encs, pass_offset = self.contrastive_parallel_all_gather(
            forward_output["pass_encs"]
        )

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = list(all_pass_encs.clone().split(config.microbatch_size))
            with self.autocast():
                micro_batch = self.model.module.encode_passages(passage).hidden
                pass_tmp[pass_offset + index] = micro_batch
                loss = self.model.module.contrastive_loss(
                    torch.cat(pass_tmp), all_rev_encs
                )
            self.deepspeed_backwards(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = list(all_rev_encs.clone().split(config.microbatch_size))
            with self.autocast():
                micro_batch = self.model.module.encode_reviews(review).hidden
                rev_tmp[rev_offset + index] = micro_batch
                loss = self.model.module.contrastive_loss(
                    all_pass_encs, torch.cat(rev_tmp)
                )
            self.deepspeed_backwards(loss)

        # Average the model gradients
        self.average_gradients()

        # Clipping
        self.clip_gradients()

        # Step the model
        self.deepspeed_step()

        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_output["forward_acc"],
        }

    def train_torch_step(
        self,
        passages: GenericBatchElement,
        reviews: GenericBatchElement,
        config: TrainConfig,
    ) -> Dict[str, TensorType[()]]:
        """
        Train the model on a single batch.
        :param passages: The passages to encode.
        :param reviews: The reviews to encode.
        :param config: The training configuration.
        :return: The loss and accuracy of the model on the batch.
        """
        with self.autocast():
            forward_output = self.model(passages, reviews, config)

        # does gradient accumulation
        self.zero_grad()

        # Encode passages in microbatches (with grad)
        for index, passage in enumerate(forward_output["pass_mbs"]):
            pass_tmp = forward_output["pass_encs"].copy()
            with self.autocast():
                pass_tmp[index] = self.model.encode_passages(passage).hidden
                loss = self.model.contrastive_loss(
                    torch.cat(pass_tmp), torch.cat(forward_output["rev_encs"])
                )

            self.torch_backwards(loss)

        # Encode reviews in microbatches (with grad)
        for index, review in enumerate(forward_output["rev_mbs"]):
            rev_tmp = forward_output["rev_encs"].copy()  # no_grad
            with self.autocast():
                rev_tmp[index] = self.model.encode_reviews(review).hidden
                # grad _just_ at positions in `index`
                loss = self.model.contrastive_loss(
                    torch.cat(forward_output["pass_encs"]), torch.cat(rev_tmp)
                )

            self.torch_backwards(loss)

        # Average the model gradients
        self.average_gradients()

        # Clipping
        self.clip_gradients()

        # Step the model
        self.torch_step()

        return {
            "Loss/Train": loss,
            "Acc/Forward": forward_output["forward_acc"],
            "Acc/Top_5_Forward": forward_output["top_k_acc"],
            "Model/logit_scale": self.model.logit_scale.sum(),
        }

    # TODO
    #def construct_tokenizer(self, )
