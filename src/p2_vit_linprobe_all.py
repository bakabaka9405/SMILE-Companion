from typing import Any

import torch
import daisy
from daisy.dataset.dataset_sample import balanced_sample
from daisy.model.mae import create_vit_model, load_mae_pretrained_weights
from common import LABELED_DATA_DIR, LABELED_DATA_SHEET, DATA_SPLIT_SHEET, NUM_CLASSES, OUTPUT_ROOT, load_smile

TRAIN_SIZE = 2000


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	checkpoint_path = OUTPUT_ROOT / 'p1_mae_pretrain' / 'checkpoint_latest.pth'
	for task in range(1, 8):
		model: Any = create_vit_model(
			'vit_base_patch16',
			num_classes=NUM_CLASSES[task],
			global_pool='avg',
			drop_path_rate=0.1,
			img_size=224,
		).to(device)
		load_mae_pretrained_weights(
			model,
			checkpoint_path=str(checkpoint_path),
			init_head=True,
			head_init_std=0.01,
			verbose=False,
		)

		for param in model.parameters():
			param.requires_grad = False

		assert isinstance(model.head, torch.nn.Linear)
		model.head = torch.nn.Sequential(
			torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
			model.head,
		)

		for param in model.head.parameters():
			param.requires_grad = True

		train_dataset, val_dataset, test_dataset = load_smile(
			root=LABELED_DATA_DIR,
			class_sheet=LABELED_DATA_SHEET,
			split_sheet=DATA_SPLIT_SHEET,
			task=task,
		)
		train_dataset = balanced_sample(train_dataset, TRAIN_SIZE, NUM_CLASSES[task])[0]
		output_dir = OUTPUT_ROOT / 'p2_vit_linprobe' / f'task_{task}'

		daisy.classifier_trainer.train_classifier(
			device=device,
			model=model,
			dataset=(train_dataset, val_dataset),
			blr=0.1,
			num_classes=NUM_CLASSES[task],
			num_workers=8,
			batch_size=256,
			epochs=50,
			smoothing=0,
			train_transform=daisy.util.transform.get_stretch_linprobe_transform(),
			val_transform=daisy.util.transform.get_stretch_val_transform(),
			warmup_epochs=5,
			save_freq=0,
			save_path=output_dir,
			log_dir=output_dir,
			print_freq=9999,
		)

		model.load_state_dict(torch.load(output_dir / 'best_model.pth', map_location='cpu'))
		y_true, _, y_output = daisy.classifier_trainer.fast_eval(
			device=device,
			model=model,
			dataset=test_dataset,
			transform=daisy.util.transform.get_stretch_val_transform(),
		)
		auc = daisy.metrics.auroc_score(y_true, y_output, num_classes=NUM_CLASSES[task], mode='macro')
		print(f'Task {task} - AUC: {auc:.4f}')


if __name__ == '__main__':
	main()
