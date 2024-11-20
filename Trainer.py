import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Any
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	classification_report,
)
from matplotlib import pyplot as plt
import os


@dataclass
class TrainParameters:
	device: torch.device
	model: Any
	epochs: int
	lr: float
	num_classes: int
	train_loader: DataLoader
	test_loader: DataLoader
	criterion: torch.nn.Module
	optimizer: torch.optim.Optimizer
	batch_size: int
	model_name: str
	pretrained: bool
	save_path_root: str


def Mkdir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path)


def CalcScore(labels, preditions) -> tuple[float, float, float, float]:
	acc = accuracy_score(labels, preditions, normalize=True)
	precision = precision_score(labels, preditions, average='macro', zero_division=1)
	recall = recall_score(labels, preditions, average='macro', zero_division=1)
	f1 = f1_score(labels, preditions, average='macro', zero_division=1)
	return float(acc), float(precision), float(recall), float(f1)


def Train(param: TrainParameters) -> None:
	all_loss: list[float] = []
	all_acc: list[float] = []
	all_precision: list[float] = []
	all_recall: list[float] = []
	all_f1: list[float] = []
	best_f1: float = 0.0
	best_state = None
	best_labels = []
	best_preds = []

	def save_score(loss: float, acc: float, precision: float, recall: float, f1: float):
		all_loss.append(loss)
		all_acc.append(acc)
		all_precision.append(precision)
		all_recall.append(recall)
		all_f1.append(f1)

	def print_score(epoch: int, loss: float, acc: float, precision: float, recall: float, f1: float):
		print(f'Epoch {epoch+1}/{param.epochs}, Loss: {loss:.6f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

	def update_best(f1: float, labels, preds):
		nonlocal best_f1, best_state, best_labels, best_preds
		if f1 <= best_f1:
			return
		best_f1 = f1
		best_labels = labels.copy()
		best_preds = preds.copy()
		# best_state = param.model.state_dict()

	def create_plot(save_path: str):
		fig, axs = plt.subplots(1, 5, figsize=(30, 5))
		axs[0].plot(all_loss, label='Loss')
		axs[0].set_title('Loss')
		axs[0].set_yscale('log')
		axs[1].plot(all_acc, label='Accuracy')
		axs[1].set_title('Accuracy')
		axs[2].plot(all_precision, label='Precision')
		axs[2].set_title('Precision')
		axs[3].plot(all_recall, label='Recall')
		axs[3].set_title('Recall')
		axs[4].plot(all_f1, label='F1')
		axs[4].set_title('F1')
		for ax in axs.flat:
			ax.legend()
		plt.tight_layout()
		plt.savefig(save_path)
		plt.close()

	def get_task_name():
		return f'{param.model_name}_{int(param.pretrained)}_{param.lr}_{param.batch_size}'

	def save_file():
		Mkdir(param.save_path_root)
		task_name = get_task_name()
		repo_dir = os.path.join(param.save_path_root, 'repo')
		# pth_dir = os.path.join(param.save_path_root, 'pth')
		plot_dir = os.path.join(param.save_path_root, 'plot')
		Mkdir(repo_dir)
		# Mkdir(pth_dir)
		Mkdir(plot_dir)
		repo_path = os.path.join(repo_dir, f'{task_name}.txt')
		# pth_path = os.path.join(pth_dir, f'{task_name}.pth')
		plot_path = os.path.join(plot_dir, f'{task_name}.png')
		# torch.save(best_state, pth_path)
		create_plot(plot_path)
		with open(repo_path, 'w') as f:
			f.write(str(classification_report(best_labels, best_preds, digits=4, zero_division=1)))

	print('Training started')
	for i in range(param.epochs):
		loss = 0.0
		param.model.train()
		for inputs, labels in param.train_loader:
			# inputs, labels = inputs.to(param.device), labels.to(param.device)
			labels = labels.to(param.device)
			param.optimizer.zero_grad()
			outputs = param.model(inputs)
			loss_m = param.criterion(outputs, labels)
			loss_m.backward()
			param.optimizer.step()
			loss += loss_m.item()
		y_true = []
		y_pred = []
		with torch.no_grad():
			param.model.eval()
			correct = 0
			total = 0
			for inputs, labels in param.test_loader:
				# inputs, labels = inputs.to(param.device), labels.to(param.device)
				labels = labels.to(param.device)
				outputs = param.model(inputs)
				predicted = torch.argmax(outputs, 1)
				correct += (predicted == labels).sum().item()
				total += labels.size(0)
				y_true.extend(labels.cpu().numpy())
				y_pred.extend(predicted.cpu().numpy())
		acc, precision, recall, f1 = CalcScore(y_true, y_pred)
		save_score(loss, acc, precision, recall, f1)
		update_best(f1, y_true, y_pred)
		print_score(i, loss, acc, precision, recall, f1)
	save_file()
