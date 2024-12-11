import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
import pandas


class ImageDataset:
	labels: list[int]
	image_tensors: list[torch.Tensor]
	device: torch.device

	def __init__(self, device: torch.device, root_dir: str):
		self.labels = []
		self.image_tensors = []
		self.device = device
		if len(root_dir) == 0:
			return
		root_files = os.listdir(root_dir)
		dataset_dir = root_dir
		if any([i.endswith('.csv') for i in root_files]):  # labeled by csv
			csv_path = ''
			for file in os.listdir(root_dir):
				if file.endswith('.csv'):
					csv_path = os.path.join(root_dir, file)
				elif os.path.isdir(os.path.join(root_dir, file)):
					dataset_dir = os.path.join(root_dir, file)
			self.image_tensors = []
			self.labels = []
			label_dict: dict[str, int] = {}
			df = pandas.read_csv(csv_path)
			for index, row in df.iterrows():
				image_path = os.path.join(dataset_dir, row['Image'])
				if not os.path.exists(image_path):
					continue

				self.image_tensors.append(decode_image(image_path, mode=ImageReadMode.RGB).to(device))
				if row['Label'] not in label_dict:
					label_dict[row['Label']] = len(label_dict)
				self.labels.append(label_dict[row['Label']])
		else:  # labeled by folder
			for label, class_name in enumerate(root_files):
				if not os.path.isdir(os.path.join(root_dir, class_name)):
					continue
				for image_path in os.listdir(os.path.join(root_dir, class_name)):
					self.image_tensors.append(decode_image(os.path.join(root_dir, class_name, image_path), mode=ImageReadMode.RGB).to(device))
					self.labels.append(label)

	def __getitem__(self, idx: int):
		return self.image_tensors[idx], self.labels[idx]


class TrainingSubset(Dataset):
	labels: list[int]
	buffer: list[torch.Tensor]
	device: torch.device

	def __init__(self, dataset: ImageDataset, indices: list[int], train_transform: transforms.Compose):
		self.labels = [dataset.labels[i] for i in indices]
		self.buffer = [dataset.image_tensors[i] for i in indices]
		self.transform = train_transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx: int):
		return self.transform(self.buffer[idx]), self.labels[idx]


class ValidationSubSet(Dataset):
	labels: list[int]
	buffer: list[torch.Tensor]
	device: torch.device

	def __init__(self, dataset: ImageDataset, indices: list[int], val_transform: transforms.Compose):
		self.labels = [dataset.labels[i] for i in indices]
		self.buffer = [val_transform(dataset.image_tensors[i]) for i in indices]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx: int):
		return self.buffer[idx], self.labels[idx]
