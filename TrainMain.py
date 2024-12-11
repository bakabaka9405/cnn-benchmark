import BatchTrain
import Util


def main():
	device = Util.GetDevice()
	print('device:', device)

	lr_lst = [0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
	batch_size_lst = [128, 16, 32, 64]
	pretrained_lst = [True]
	model_lst = [
		'timm:resnet34',
	]
	epochs = 200

	param1 = BatchTrain.BatchTrainParameters(
		dataset_root=r'C:\Resources\Datasets\Smile\微笑弧线_crop',
		save_path_root=r'C:\Temp\Benchmark',
		task_name='微笑弧线',
		device=device,
		epochs=epochs,
		lr_lst=lr_lst,
		batch_size_lst=batch_size_lst,
		pretrained_lst=pretrained_lst,
		model_lst=model_lst,
		num_classes=3,
	)

	param2 = BatchTrain.BatchTrainParameters(
		dataset_root='data',
		save_path_root='result',
		task_name='笑线高度',
		device=device,
		epochs=epochs,
		lr_lst=lr_lst,
		batch_size_lst=batch_size_lst,
		pretrained_lst=pretrained_lst,
		model_lst=model_lst,
		num_classes=4,
	)

	param3 = BatchTrain.BatchTrainParameters(
		dataset_root='data',
		save_path_root='result',
		task_name='笑线高度',
		device=device,
		epochs=epochs,
		lr_lst=lr_lst,
		batch_size_lst=batch_size_lst,
		pretrained_lst=pretrained_lst,
		model_lst=model_lst,
		num_classes=3,
	)

	BatchTrain.BatchTrain(param1)
	BatchTrain.BatchTrain(param2)
	BatchTrain.BatchTrain(param3)


if __name__ == '__main__':
	main()
