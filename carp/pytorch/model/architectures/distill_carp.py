import torch
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from carp.configs import CARPConfig, ModelConfig, TrainConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.model.encoders import get_encoder
from carp.util import mbTokens, generate_indices
from typing import List

from carp.pytorch.data.utils.data_util import BatchElement
from carp.pytorch.data.distill_pipeline import DistillBatchElement

patch_typeguard()

@typechecked
@register_architecture
class DistillCARP(BaseModel):
	def __init__(self, config: ModelConfig):
		super().__init__(config)

	def _reduction(self, logits: TensorType["pass_N", "rev_N"]) -> TensorType["pass_N", "reduced_rev_N"]:
		n = logits.shape[0]
		logits = torch.sum(logits.reshape((n,-1,self.reviews_per_passage)), dim=-1)
		return logits

	def compute_accuracy(self, x: TensorType[-1, "latent_dim"], y: TensorType[-1, "latent_dim"]):
		with torch.no_grad():
			n = x.shape[0]
			x = F.normalize(x)
			y = F.normalize(y)
			logits = x @ y.T * self.logit_scale.exp()

			logits = self._reduction(logits)

			labels = torch.arange(n, device=self.config.device)
			acc_i = (torch.argmax(logits, dim=1) == labels).sum()
			acc_t = (torch.argmax(logits, dim=0) == labels).sum()
		return (acc_i + acc_t) / n / 2

	def contrastive_loss(
		self, x: TensorType[-1, "latent_dim"], y: TensorType[-1, "latent_dim"]
	) -> TensorType[(), float]:

		n = x.shape[0]
		# small term added to avoid nans in low precision softmax
		#(num_passages, num_reviews)
		logits = self.cosine_sim(x,y) * self.logit_scale.exp()
		logits_i = F.softmax(logits, dim=-1)
		logits_t = F.softmax(logits, dim=0)
		reduced_logits_i = self._reduction(logits_i)
		reduced_logits_t = self._reduction(logits_t)
		#Reduce logits into diagonal
		labels = torch.arange(n, device=self.config.device)
		loss_i = F.nll_loss(torch.log(reduced_logits_i), labels)
		loss_t = F.nll_loss(torch.log(reduced_logits_t.T), labels)
		print(loss_i, loss_t)
		return (loss_i + loss_t) / 2


	def train_step(
		self,
		passages: BatchElement,
		reviews: DistillBatchElement,
		config: TrainConfig,
		opt: torch.optim.Optimizer,
		scaler: torch.cuda.amp.GradScaler,
	) -> Dict[str, TensorType[()]]:

		print('train_step reviews', reviews.input_ids.size())
		self.reviews_per_passage = reviews.input_ids.size()[0] // passages.input_ids.size()[0]

		microbatch_inds_passages = generate_indices(
			passages.input_ids.shape[0], config.microbatch_size, shuffle=False
		)
		microbatch_inds_reviews = generate_indices(
			reviews.input_ids.shape[0], config.microbatch_size, shuffle=False
		)
		# Split tokens and masks into these microbatches
		pass_mbs: List[BatchElement] = [
			BatchElement(passages.input_ids[i], passages.mask[i]) for i in microbatch_inds_passages
		]
		rev_mbs: List[DistillBatchElement] = [
			DistillBatchElement(reviews.input_ids[i], reviews.mask[i], reviews.reviews_per_passage) for i in microbatch_inds_reviews
		]

		reviews_per_passage = reviews.reviews_per_passage

		# Initially get all encodings without grad
		pass_encs, rev_encs = self.calculate_embeddings(pass_mbs, rev_mbs)

		print('rev_encs size', len(rev_encs))

		#compute accuracy
		forward_acc = self.compute_accuracy(torch.cat(pass_encs), torch.cat(rev_encs))

		# does gradient accumulation
		self.zero_grad(opt)

		# Encode passages in microbatches (with grad)
		for index, passage in enumerate(pass_mbs):
			pass_tmp = pass_encs.copy()
			with torch.cuda.amp.autocast():
				pass_tmp[index] = self.encode_passages(passage).hidden
				loss  = self.contrastive_loss(
					torch.cat(pass_tmp), torch.cat(rev_encs)
				)
			scaler.scale(loss).backward()
		# Encode reviews in microbatches (with grad)
		for index, review in enumerate(rev_mbs):
			rev_tmp = rev_encs.copy()  # no_grad
			with torch.cuda.amp.autocast():
				rev_tmp[index] = self.encode_reviews(review).hidden
				# grad _just_ at positions in `index`
				loss = self.contrastive_loss(
					torch.cat(pass_encs), torch.cat(rev_tmp)
				)
			scaler.scale(loss).backward()
		# Clipping
		if self.config.grad_clip != -1:
			scaler.unscale_(opt)
			torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

		self.step(scaler, opt)
		return {
			"Loss/Train": loss,
			"Acc/Forward": forward_acc,
		}
