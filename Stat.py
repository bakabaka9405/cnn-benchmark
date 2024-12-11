import pandas as pd
import os

root = r'C:\Temp\Benchmark\微笑弧线_crop\repo'
dst = r'C:\Temp\Benchmark\微笑弧线_crop.xlsx'

model_lst = [
	'resnet18',
	'resnet34',
	'resnet50',
	'resnet101',
	'resnet152',
	'rexnet_100',
	'rexnet_130',
	'rexnet_150',
	'rexnet_200',
	'rexnet_300',
	'resnext50_32x4d',
	'resnext101_32x4d',
	'resnext101_64x4d',
	'wide_resnet50_2',
	'wide_resnet101_2',
	'seresnext50_32x4d',
	'seresnext101_32x4d',
	'efficientnet_b4',
	'efficientnet_b5',
	'densenet121',
	'densenet161',
	'densenet169',
	'densenet201',
]

def parse_file(file: str):
	file=file[:file.rfind('.')]
	i = file.rfind('_')
	batch_size = int(file[i + 1 :])
	file = file[:i]
	i = file.rfind('_')
	lr = float(file[i + 1 :])
	file = file[:i]
	i = file.rfind('_')
	pretrained = int(file[i + 1 :])
	file = file[:i]
	i = file.rfind('_')
	model = file
	return model, pretrained, batch_size, lr


def main():
	data = []
	# walk
	for dirpath, dirnames, filenames in os.walk(root):
		for filename in filenames:
			model, pretrained, batch_size, lr = parse_file(filename)
			if model not in model_lst:
				continue
			with open(os.path.join(dirpath, filename), 'r') as f:
				lines = f.readlines()
				acc=float(lines[6].split()[1])
				data.append({'model': model, 'pretrained': pretrained, 'batch_size': batch_size, 'lr': lr, 'acc': acc})
	df = pd.DataFrame(data)
	df.to_excel(dst, index=False)

if __name__ == '__main__':
	main()
