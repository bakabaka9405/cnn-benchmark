from itertools import product
from Util import GetModel, LoadDataset, GetDataLoader, GetTrainTransform, GetValTransform
from KFold import KFoldGenerator
import Trainer
import torch
import gc
from dataclasses import dataclass


@dataclass
class BatchTrainParameters:
	dataset_root: str
	save_path_root: str
	task_name: str
	device: torch.device
	epochs: int
	lr_lst: list[float]
	batch_size_lst: list[int]
	pretrained_lst: list[bool]
	model_lst: list[str]
	num_classes: int


def BatchTrain(param: BatchTrainParameters) -> None:
	print('task: ', param.task_name)
	dataset = LoadDataset(param.device, param.dataset_root)
	print('dataset loaded')
	criterion = torch.nn.CrossEntropyLoss()
	gen = KFoldGenerator(dataset, 5, train_transform=GetTrainTransform(), val_transform=GetValTransform())
	for pretrained, batch_size, lr, model_name, (fold, (train_set, val_set)) in product(
		param.pretrained_lst, param.batch_size_lst, param.lr_lst, param.model_lst, enumerate(gen)
	):
		print(f'model: {model_name}, pretrained: {pretrained}, batch_size: {batch_size}, lr: {lr}, fold: {fold+1}')
		train_loader, val_loader = GetDataLoader(train_set, val_set, batch_size, batch_size)
		print('DataLoader ready')

		try:
			model = None
			gc.collect()
			torch.cuda.empty_cache()
			model = GetModel(model_name, param.num_classes, pretrained)
			model.to(param.device)
		except Exception as e:
			print(f'Error: {e}, skipped')
			continue
		print('model ready')

		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

		train_param = Trainer.TrainParameters(
			device=param.device,
			model=model,
			epochs=param.epochs,
			fold=fold + 1,
			lr=lr,
			num_classes=param.num_classes,
			train_loader=train_loader,
			test_loader=val_loader,
			criterion=criterion,
			optimizer=optimizer,
			batch_size=batch_size,
			model_name=model_name[5:] if model_name.startswith('timm:') else model_name,
			pretrained=pretrained,
			save_path_root=f'{param.save_path_root}\\{param.task_name}',
		)

		Trainer.Train(train_param)
