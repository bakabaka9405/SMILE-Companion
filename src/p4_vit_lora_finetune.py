import torch
import daisy
from daisy.model.mae import create_vit_model, load_mae_pretrained_weights
from peft import LoraConfig, get_peft_model
from common import LABELED_DATA_DIR, LABELED_DATA_SHEET, DATA_SPLIT_SHEET, NUM_CLASSES, OUTPUT_ROOT, load_smile


def load_lora_vit(model):
	lora_cfg = LoraConfig(
		r=8,
		lora_alpha=16,
		target_modules=['qkv', 'proj', 'fc1', 'fc2'],
		modules_to_save=['head'],
		lora_dropout=0.1,
		bias='none',
	)
	return get_peft_model(model, lora_cfg)


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	checkpoint_path = OUTPUT_ROOT / 'p1_mae_pretrain' / 'checkpoint_latest.pth'
	for task in range(1, 8):
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
		model = load_lora_vit(model).to(device)

		train_dataset, val_dataset, test_dataset = load_smile(
			root=LABELED_DATA_DIR,
			class_sheet=LABELED_DATA_SHEET,
			split_sheet=DATA_SPLIT_SHEET,
			task=task,
		)
		output_dir = OUTPUT_ROOT / 'p4_vit_lora_finetune' / f'task_{task}'

		daisy.mae_finetune.mae_finetune(
			device=device,
			model=model,
			dataset=(train_dataset, val_dataset),
			blr=1e-4,
			num_classes=NUM_CLASSES[task],
			num_workers=6,
			batch_size=64,
			epochs=50,
			train_transform=daisy.util.transform.get_stretch_train_transform(),
			val_transform=daisy.util.transform.get_stretch_val_transform(),
			warmup_epochs=5,
			save_freq=0,
			save_path=output_dir,
			log_dir=output_dir,
			pin_memory=False,
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
