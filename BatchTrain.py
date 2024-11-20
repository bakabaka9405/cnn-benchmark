from itertools import product
from Util import GetDevice, GetModel, LoadDataset, GetDataLoader
import Trainer
import torch

dataset_root = r'C:\Resources\Datasets\Smile\笑线高度'
save_path_root = r'C:/Temp/Benchmark/'
lr_lst = [0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
batch_size_lst = [128, 16, 32, 64]
pretrained_lst = [True, False]
model_lst = [
	'timm:resnet18',
	'timm:resnet34',
	'timm:resnet50',
	'timm:resnet101',
	'timm:resnet152',
	'timm:rexnet_100',
	'timm:rexnet_130',
	'timm:rexnet_150',
	'timm:rexnet_200',
	'timm:rexnet_300',
	'timm:resnext50_32x4d',
	'timm:resnext50d_32x4d',
	'timm:resnext101_32x4d',
	'timm:resnext101_32x8d',
	'timm:resnext101_32x16d',
	'timm:resnext101_32x32d',
	'timm:resnext101_64x4d',
	'timm:wide_resnet50_2',
	'timm:wide_resnet101_2',
	'timm:seresnext50_32x4d',
	'timm:seresnext101_32x4d',
	'timm:seresnext101_32x8d',
	'timm:seresnext101_64x4d',
	'timm:efficientnet_b4',
	'timm:efficientnet_b5',
	'timm:pvt_v2_b4',
	'timm:pvt_v2_b5',
	'timm:densenet121',
	'timm:densenet161',
	'timm:densenet169',
	'timm:densenet201',
	'timm:mobilenetv3_large_100',
	'timm:mobilenetv3_rw',
	'timm:mobilenetv4_conv_large',
	'timm:mobilenetv4_hybrid_large',
]
num_classes = 3
epochs = 200


def DownloadModel(model_name: str):
	GetModel(model_name, num_classes, True)


def main():
	device = GetDevice()
	print('device:', device)
	dataset = LoadDataset(device, dataset_root)
	print('dataset loaded')
	criterion = torch.nn.CrossEntropyLoss()
	for pretrained, batch_size, lr, model_name in product(pretrained_lst, batch_size_lst, lr_lst, model_lst):
		print(f'model: {model_name}, pretrained: {pretrained}, batch_size: {batch_size}, lr: {lr}')

		try:
			model = GetModel(model_name, num_classes, pretrained)
			model.to(device)
		except Exception as e:
			print(f'Error: {e}, skipped')
			continue
		print('model ready')

		train_loader, val_loader = GetDataLoader(dataset, batch_size, 1000)
		print('DataLoader ready')

		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

		param = Trainer.TrainParameters(
			device=device,
			model=model,
			epochs=epochs,
			lr=lr,
			num_classes=num_classes,
			train_loader=train_loader,
			test_loader=val_loader,
			criterion=criterion,
			optimizer=optimizer,
			batch_size=batch_size,
			model_name=model_name[5:] if model_name.startswith('timm:') else model_name,
			pretrained=pretrained,
			save_path_root=f'{save_path_root}\\{dataset_root[dataset_root.replace('\\', '/').rfind("/")+1:]}',
		)

		Trainer.Train(param)


if __name__ == '__main__':
	main()
