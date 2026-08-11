from itertools import product

import torch
import daisy
from daisy.model.mae import create_vit_model, load_mae_pretrained_weights
from common import LABELED_DATA_DIR, LABELED_DATA_SHEET, DATA_SPLIT_SHEET, NUM_CLASSES, OUTPUT_ROOT, load_smile

LR = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	checkpoint_path = OUTPUT_ROOT / 'p1_mae_pretrain' / 'checkpoint_latest.pth'
	for lr, task in product(LR, range(1, 8)):
		model = create_vit_model(
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
			verbose=False,
		)

		train_dataset, val_dataset, _ = load_smile(
			root=LABELED_DATA_DIR,
			class_sheet=LABELED_DATA_SHEET,
			split_sheet=DATA_SPLIT_SHEET,
			task=task,
		)
		output_dir = OUTPUT_ROOT / 'p3_vit_finetune' / f'{lr}' / f'task_{task}'

		daisy.mae_finetune.mae_finetune(
			device=device,
			model=model,
			dataset=(train_dataset, val_dataset),
			blr=lr,
			num_classes=NUM_CLASSES[task],
			num_workers=(6, 2),
			batch_size=64,
			epochs=50,
			train_transform=daisy.util.transform.get_stretch_train_transform(),
			val_transform=daisy.util.transform.get_stretch_val_transform(),
			warmup_epochs=5,
			save_freq=0,
			save_path=output_dir,
			log_dir=output_dir,
			pin_memory=True,
			mixup=0,
			cutmix=0,
			early_stop=True,
		)


if __name__ == '__main__':
	main()
