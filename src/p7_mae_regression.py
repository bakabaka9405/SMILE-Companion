from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from daisy.metrics import evaluate_regression_scores
from daisy.model.mae.models_vit import create_vit_model, load_mae_pretrained_weights
from daisy.training import BestModel, CSVLog, CosineAnnealingLR, EpochPrint, Eval, Trainer
from daisy.util.transform import get_stretch_train_transform, get_stretch_val_transform
from regression_util import DataLoader, OUTPUT_ROOT, load_datasets, save_predictions


def build_model(mae_checkpoint: Path, device: torch.device) -> nn.Sequential:
	backbone = create_vit_model(
		'vit_base_patch16',
		global_pool='avg',
		num_classes=0,
		drop_path_rate=0.1,
		img_size=224,
	)
	load_mae_pretrained_weights(backbone, str(mae_checkpoint), init_head=False, verbose=False)
	model = nn.Sequential(backbone, nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
	return model.to(device)


def train(
	epochs: int,
	batch_size: int,
	blr: float,
	seed: int,
	device: torch.device,
	num_workers: int,
	mae_checkpoint: Path,
) -> None:
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	train_dataset, val_dataset, test_dataset = load_datasets(
		seed,
		get_stretch_train_transform(),
		get_stretch_val_transform(),
	)
	print(f'Data loaded  train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}')

	output_dir = OUTPUT_ROOT / 'p7_mae_regression'
	output_dir.mkdir(parents=True, exist_ok=True)
	model = build_model(mae_checkpoint, device)
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=True,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False,
	)

	lr = blr * batch_size / 256.0
	optimizer = torch.optim.AdamW(
		[
			{'params': model[0].parameters(), 'lr_scale': 0.1},
			{'params': model[1:].parameters()},
		],
		lr=lr,
		weight_decay=0.05,
	)
	evaluator = Eval(eval_fn=lambda trainer: evaluate_regression_scores(trainer.inference(val_loader), val_dataset.labels))
	best_model = BestModel(evaluator, watch_metric='pc', mode='max', save_path=output_dir)
	trainer = (
		Trainer(model, optimizer, nn.MSELoss(), device, use_amp=True)
		.after_forward(lambda state, outputs: outputs.squeeze(1))
		.use(
			CosineAnnealingLR(lr=lr, warmup_epochs=5),
			evaluator,
			best_model,
			EpochPrint(evaluator, metric_names=['pc', 'mae', 'rmse', 'icc_a1']),
			CSVLog(output_dir / 'history.csv', evaluator),
		)
	)

	started_at = time.perf_counter()
	trainer.fit(train_loader, epochs=epochs)
	model.load_state_dict(torch.load(output_dir / 'best_model.pth', map_location='cpu', weights_only=True))
	model.to(device)

	scores = trainer.inference(test_loader)
	metrics = evaluate_regression_scores(scores, test_dataset.labels)
	save_predictions(
		output_dir,
		test_dataset.file_paths,
		np.asarray(test_dataset.labels, dtype=np.float32),
		scores.float().numpy(),
	)
	print(f'Training time: {time.perf_counter() - started_at:.1f}s  best_epoch={best_model.best_epoch + 1}')
	print(
		f'Test: PC={metrics["pc"]:.6f}  MAE={metrics["mae"]:.6f}  '
		f'RMSE={metrics["rmse"]:.6f}  ICC(A,1)={metrics["icc_a1"]:.6f}'
	)


def main() -> None:
	parser = argparse.ArgumentParser(description='MAE-pretrained ViT aesthetic-score regression')
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--blr', type=float, default=1e-3)
	parser.add_argument('--seed', type=int, default=3407)
	parser.add_argument('--device', type=str, default=None)
	parser.add_argument('--num-workers', type=int, default=4)
	parser.add_argument('--mae-checkpoint', type=Path, default=OUTPUT_ROOT / 'p1_mae_pretrain' / 'checkpoint_latest.pth')
	args = parser.parse_args()
	device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train(args.epochs, args.batch_size, args.blr, args.seed, device, args.num_workers, args.mae_checkpoint)


if __name__ == '__main__':
	main()
