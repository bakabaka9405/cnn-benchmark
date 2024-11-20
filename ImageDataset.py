import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
import pandas
import random
import threading


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

		if any([i.endswith('.csv') for i in root_files]):  # labeled by csv
			csv_path = ''
			for file in os.listdir(root_dir):
				if file.endswith('.csv'):
					csv_path = os.path.join(root_dir, file)
				elif os.path.isdir(os.path.join(root_dir, file)):
					self.root_dir = os.path.join(root_dir, file)
			self.image_tensors = []
			self.labels = []
			label_dict: dict[str, int] = {}
			df = pandas.read_csv(csv_path)
			for index, row in df.iterrows():
				image_path = os.path.join(root_dir, row['Image'])
				if not os.path.exists(image_path):
					continue
				# assert image_path.endswith(".jpg")

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

	def random_split(self, train_transform: transforms.Compose, val_transform: transforms.Compose, ratio: float):
		index = [i for i in range(len(self.labels))]
		random.shuffle(index)

		train_size = int(ratio * len(self.labels))

		class TrainingSubSet(Dataset):
			def __init__(self, dataset: ImageDataset, indices: list[int], train_transform: transforms.Compose):
				self.dataset = dataset
				self.indices = indices
				self.transform = train_transform
				self.buffer: list[list[tuple[torch.Tensor, int]]] = [[] for i in range(len(indices))]
				# self.lock = [threading.Lock() for i in range(len(indices))]
				self.thread: threading.Thread | None = None
				self.thread_end = False

			def __len__(self):
				return len(self.indices)

			def __getitem__(self, idx: int):
				if len(self.buffer[idx]) != 0:
					return self.buffer[idx].pop()

				image, label = self.dataset[self.indices[idx]]
				return self.transform(image), label

			def start_make_buffer(self):
				def make_buffer():
					p = 0
					while not self.thread_end:
						if p >= len(self.indices):
							p = 0
						if len(self.buffer[p]) >= 10:
							p = p + 1
							continue
						image, label = self.dataset[self.indices[p]]
						self.buffer[p].append((self.transform(image), label))
						p = p + 1

				self.thread = threading.Thread(target=make_buffer)
				self.thread.start()

			def stop_thread(self):
				self.thread_end = True
				if self.thread is not None:
					self.thread.join()

		class ValidationSubSet(Dataset):
			labels: list[int]
			buffer: list[torch.Tensor]

			def __init__(self, dataset: ImageDataset, indices: list[int], val_transform: transforms.Compose):
				self.labels = [dataset.labels[i] for i in indices]
				self.buffer = [val_transform(dataset.image_tensors[i]) for i in indices]

			def __len__(self):
				return len(self.labels)

			def __getitem__(self, idx: int):
				return self.buffer[idx], self.labels[idx]

		return TrainingSubSet(self, index[:train_size], train_transform), ValidationSubSet(self, index[train_size:], val_transform)
