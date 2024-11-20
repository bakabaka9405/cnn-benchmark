from typing import Any as any
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


import timm
from ImageDataset import ImageDataset


def GetDevice():
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ZeroOneNormalize:
	def __call__(self, tensor: torch.Tensor):
		return tensor.float().div(255)


def GetTrainTransform():
	return transforms.Compose(
		[
			transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
			transforms.Resize((128, 128)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.691208, 0.50378054, 0.45438778], [0.14654742, 0.14745905, 0.14166589]),
		]
	)


def GetValTransform():
	return transforms.Compose(
		[
			transforms.Resize((128, 128)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.691208, 0.50378054, 0.45438778], [0.14654742, 0.14745905, 0.14166589]),
		]
	)


def LoadDataset(device: torch.device, dataset_root: str):
	dataset = ImageDataset(device=device, root_dir=dataset_root)
	return dataset


def GetDataLoader(dataset: ImageDataset, train_batchsize: int, test_batchsize: int, ratio: float = 0.8):
	train_set, test_set = dataset.random_split(GetTrainTransform(), GetValTransform(), ratio)
	train_loader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=test_batchsize, shuffle=False)
	return train_loader, test_loader


def GetModel(model_name: str, num_classes: int, pretrained: bool = False):
	model_class: any
	if model_name == 'densenet121':
		model_class = torchvision.models.densenet121
	elif model_name == 'shufflenet_v2_x1_0':
		model_class = torchvision.models.shufflenet_v2_x1_0
	elif model_name == 'resnet18':
		model_class = torchvision.models.resnet18
	elif model_name.startswith('timm:'):
		model_name = model_name[5:]

		def create(num_classes: int = 1000, pretrained: bool = False):
			return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

		model_class = create
	else:
		raise ValueError('Invalid model type')
	if pretrained:
		model = model_class(pretrained=True)
		# Change the output layer to match the number of classes
		# The output layer is either model.fc or model.head.fc or model.classifier
		# Differ between models created by torchvision or timm
		if hasattr(model, 'fc'):
			model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
		elif hasattr(model, 'head'):
			if not hasattr(model.head, 'fc'):
				model.head = torch.nn.Linear(model.head.in_features, num_classes)
			else:
				model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes)
		elif hasattr(model, 'classifier'):
			model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
		else:
			raise ValueError('Invalid model type')
		return model
	else:
		# Use Random initialization
		return model_class(num_classes=num_classes)
