import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
import pandas
import random


class ImageDataset(Dataset):
	labels: list[int]
	buffer: list[torch.Tensor]
	transform: transforms.Compose | None
	preloaded: bool

	def __init__(self, device: torch.device | None, root_dir: str):
		self.labels = []
		self.buffer = []
		self.transform = None
		self.preloaded = False
		if len(root_dir) == 0:
			return
		root_files = os.listdir(root_dir)

		if any([i.endswith('.csv') for i in root_files]):  # labeled by csv
			csv_path = ''
			for file in os.listdir(root_dir):
				if file.endswith('.csv'):
					csv_path = os.path.join(root_dir, file)
				elif os.path.isdir(os.path.join(root_dir, file)):
					self.root_dir = os.path.join(root_dir, file)
			self.buffer = []
			self.labels = []
			label_dict: dict[str, int] = {}
			df = pandas.read_csv(csv_path)
			for index, row in df.iterrows():
				image_path = os.path.join(root_dir, row['Image'])
				if not os.path.exists(image_path):
					continue
				# assert image_path.endswith(".jpg")

				self.buffer.append(decode_image(image_path, mode=ImageReadMode.RGB).to(device))
				if row['Label'] not in label_dict:
					label_dict[row['Label']] = len(label_dict)
				self.labels.append(label_dict[row['Label']])
		else:  # labeled by folder
			for label, class_name in enumerate(root_files):
				if not os.path.isdir(os.path.join(root_dir, class_name)):
					continue
				for image_path in os.listdir(os.path.join(root_dir, class_name)):
					self.buffer.append(decode_image(os.path.join(root_dir, class_name, image_path), mode=ImageReadMode.RGB).to(device))
					self.labels.append(label)

	def __len__(self):
		return len(self.buffer)

	def __getitem__(self, idx: torch.Tensor | int):
		if torch.is_tensor(idx):
			idx = idx.tolist()  # type: ignore
		image = self.buffer[idx]
		label = self.labels[idx]
		if self.transform and not self.preloaded:
			image = self.transform(image)
		return image, label

	def preload(self):
		assert self.transform is not None
		if self.preloaded:
			return
		self.preloaded = True
		for i in range(len(self.buffer)):
			self.buffer[i] = self.transform(self.buffer[i])

	def random_split(self, ratio: float):
		# shuffle
		index = [i for i in range(len(self))]
		random.shuffle(index)
		self.buffer = [self.buffer[i] for i in index]
		self.labels = [self.labels[i] for i in index]

		# split
		train_size = int(ratio * len(self))
		train_set = ImageDataset(None, '')
		test_set = ImageDataset(None, '')
		train_set.buffer = self.buffer[:train_size]
		test_set.buffer = self.buffer[train_size:]
		train_set.labels = self.labels[:train_size]
		test_set.labels = self.labels[train_size:]
		return train_set, test_set
