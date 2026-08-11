import torch
import daisy
from daisy.classifier_trainer import fast_calc_metrics, fast_eval
from daisy.model.mae import create_vit_model
from common import LABELED_DATA_DIR, LABELED_DATA_SHEET, DATA_SPLIT_SHEET, NUM_CLASSES, OUTPUT_ROOT, load_smile


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	for task in range(1, 8):
		model = create_vit_model(
			'vit_base_patch16',
			num_classes=NUM_CLASSES[task],
			global_pool='avg',
			drop_path_rate=0.1,
			img_size=224,
		).to(device)
		model.load_state_dict(
			torch.load(
				OUTPUT_ROOT / 'p3_vit_finetune' / '0.001' / f'task_{task}' / 'best_model.pth',
				map_location='cpu',
			)
		)
		model.eval()

		test_dataset = load_smile(
			root=LABELED_DATA_DIR,
			class_sheet=LABELED_DATA_SHEET,
			split_sheet=DATA_SPLIT_SHEET,
			task=task,
		)[2]
		test_dataset.applyTransform(daisy.util.transform.get_stretch_val_transform())
		y_true, y_pred, y_outputs = fast_eval(device=device, model=model, dataset=test_dataset)
		metrics = fast_calc_metrics(y_true, y_pred, num_classes=NUM_CLASSES[task])
		auc = daisy.metrics.auroc_score(y_true, y_outputs, num_classes=NUM_CLASSES[task], mode='macro')
		print(
			f'Task {task}, Acc: {metrics.acc:.4f}, Precision: {metrics.precision:.4f}, '
			f'Recall: {metrics.recall:.4f}, F1: {metrics.f1:.4f}, AUC: {auc:.4f}, '
			f'Matrix: {metrics.confusion_matrix}'
		)


if __name__ == '__main__':
	main()
