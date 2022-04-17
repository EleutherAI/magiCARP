import ast
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from numpy.random import choice
from torchtyping import TensorType

from carp.pytorch.data import *
from carp.pytorch.model.encoders import BaseEncoder
from carp.pytorch.data.scarecrow_pipeline import ScarecrowTargetElement
import os

@register_datapipeline
class AlignmentDataPipeline(BaseDataPipeline):

	"""Dataset wrapper class to ease working with the CARP dataset and Pytorch data utilities."""

	def __init__(
		self,
		dupe_protection: bool = True,
		path: str = "dataset",
	):
		# We'll load scarecrow by default but in the future I hope to have a very standardized format
		csv = pd.read_csv(os.path.join(path,'alignment.csv'))
		self.passages = csv['story'].values.tolist()
		self.reviews = torch.tensor(csv.loc[:, csv.columns != 'story'].values.tolist())


	@staticmethod
	def tokenizer_factory(_tok: Callable, encoder: BaseEncoder) -> Callable:
		"""Function factory that creates a collate function for use with a torch.util.data.Dataloader

		Args:
			tokenizer (PreTrainedTokenizer): A Huggingface model tokenizer, taking strings to torch Tensors
			context_len (int): Max length of the passages passed to the tokenizer

		Returns:
			Callable: A function that will take a batch of string tuples and tokenize them properly.
		"""

		@typechecked
		def collate(
			data: Iterable[Tuple[str, str]]
		) -> Tuple[BatchElement, ScarecrowTargetElement]:
			passages, reviews = zip(*data)
			pass_tokens = _tok(list(passages))
			pass_masks = pass_tokens["attention_mask"]
			pass_tokens = pass_tokens["input_ids"]

			# concat all the target distributions
			reviews = torch.cat(
				list(map(lambda x: torch.tensor(x).unsqueeze(0), list(reviews))), dim=0
			)

			return (
				BatchElement(pass_tokens, pass_masks),
				ScarecrowTargetElement(reviews),
			)

		return collate
