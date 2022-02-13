from torch.functional import Tensor
from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder
from transformers.data.data_collator import DataCollatorForLanguageModeling

from dataclasses import dataclass
from torchtyping import TensorType
from typing import List
import torch

#TODO:
'''Custom chunk_batch_element
'''


@dataclass
class DistillBatchElement(BatchElement):
	#Reducing over critiques for same stories
	#reduction_matrix : TensorType["pass_N", -1]
	reviews_per_passage: int


@register_datapipeline
class DistillDataPipeline(BaseDataPipeline):

	"""Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""
	def __init__(
		self,
		#Prevents duplicates of multiple stories
		dupe_protection: bool = True,
		path: str = "dataset",
	):
		super().__init__(dupe_protection, path)

	#Overload for data format (passage, [crit_1,...,crit_n])
	def __getitem__(self, index: int) -> Tuple[str, List[str]]:
		return self.passages[index], [self.reviews[index]]*2

	@staticmethod
	def tokenizer_factory(_tok : Callable, encoder: BaseEncoder)  -> Callable:
		"""Function factory that creates a collate function for use with a torch.util.data.Dataloader

		Args:
			tokenizer (PreTrainedTokenizer): A Huggingface model tokenizer, taking strings to torch Tensors
			context_len (int): Max length of the passages passed to the tokenizer

		Returns:
			Callable: A function that will take a batch of string tuples and tokenize them properly.
		"""

		@typechecked
		def collate(
			data: Iterable[Tuple[str, List[str]]]
		) -> Tuple[BatchElement, DistillBatchElement]:
			#Expects us to double reviews beforehand: passing in list of critiques for each story
			passages, review_lists = zip(*data)
			reviews_per_passage = len(review_lists[0])
			print(reviews_per_passage)
			reviews = [review for review_list in review_lists for review in review_list]
			print(len(reviews))
			pass_tokens, rev_tokens = _tok(list(passages)), _tok(list(reviews))
			pass_masks = pass_tokens["attention_mask"]
			rev_masks = rev_tokens["attention_mask"]
			pass_tokens = pass_tokens["input_ids"]
			rev_tokens = rev_tokens["input_ids"]

			#eduction_matrix = torch.arange(0, rev_tokens.size()[0], step=1).reshape(-1, reviews_per_passage)

			return (
				BatchElement(pass_tokens, pass_masks),
				DistillBatchElement(rev_tokens, rev_masks, reviews_per_passage),
			)

		return collate