import torch
from torchvision import transforms
from ImageDataset import ImageDataset, TrainingSubset, ValidationSubSet
import random


class KFoldGenerator:
	def __init__(self, dataset: ImageDataset, k: int, train_transform: transforms.Compose, val_transform: transforms.Compose):
		self.dataset = dataset
		self.k = k
		self.indices = list(range(len(dataset.labels)))
		random.shuffle(self.indices)
		self.fold_size = len(self.indices) // k
		self.fold_indices = [self.indices[i * self.fold_size : (i + 1) * self.fold_size] for i in range(k)]
		self.fold_indices[-1] += self.indices[self.fold_size * k :]
		self.current_fold = 0
		self.train_transform = train_transform
		self.val_transform = val_transform

	def __iter__(self):
		self.current_fold = 0
		return self

	def __next__(self):
		if self.current_fold >= self.k:
			raise StopIteration
		train_indices = []
		for i in range(self.k):
			if i != self.current_fold:
				train_indices += self.fold_indices[i]
		val_indices = self.fold_indices[self.current_fold]
		self.current_fold += 1
		return TrainingSubset(self.dataset, train_indices, self.train_transform), ValidationSubSet(self.dataset, val_indices, self.val_transform)
