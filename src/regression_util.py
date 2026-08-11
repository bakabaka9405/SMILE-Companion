from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from timm.data.loader import MultiEpochsDataLoader

from daisy.dataset import DiskDataset

DataLoader = MultiEpochsDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / 'data' / 'regression' / 'images'
SCORES_FILE = PROJECT_ROOT / 'data' / 'regression' / 'scores.txt'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'


def _find_image(image_dir: Path, stem: str) -> Path:
	for ext in ('.jpg', '.png', '.jpeg', '.bmp'):
		path = image_dir / f'{stem}{ext}'
		if path.exists():
			return path
	raise FileNotFoundError(f'Image not found for stem {stem!r} in {image_dir}')


def load_all_samples() -> tuple[list[Path], list[float]]:
	if not SCORES_FILE.exists():
		raise FileNotFoundError(f'Scores file not found: {SCORES_FILE}')
	if not IMAGE_DIR.exists():
		raise FileNotFoundError(f'Image directory not found: {IMAGE_DIR}')

	scores = [float(score) for score in SCORES_FILE.read_text(encoding='utf-8').split()]
	file_paths = [_find_image(IMAGE_DIR, str(index)) for index in range(1, len(scores) + 1)]
	return file_paths, scores


def load_datasets(
	seed: int,
	train_transform: Callable[..., Any],
	eval_transform: Callable[..., Any],
) -> tuple[DiskDataset[float], DiskDataset[float], DiskDataset[float]]:
	file_paths, scores = load_all_samples()
	if len(file_paths) != 1000 or len(scores) != 1000:
		raise ValueError(f'Expected 1000 samples, got {len(file_paths)} images and {len(scores)} scores')

	indices = np.random.default_rng(seed).permutation(len(file_paths))
	train_indices, val_indices, test_indices = indices[:800], indices[800:900], indices[900:]
	if len(train_indices) != 800 or len(val_indices) != 100 or len(test_indices) != 100:
		raise AssertionError('Expected an 800/100/100 split')
	if set(train_indices) & set(val_indices) or set(train_indices) & set(test_indices) or set(val_indices) & set(test_indices):
		raise AssertionError('Dataset splits overlap')
	if len(set(train_indices) | set(val_indices) | set(test_indices)) != len(file_paths):
		raise AssertionError('Dataset splits do not cover all samples')

	train_paths = [file_paths[index] for index in train_indices]
	train_scores = [scores[index] for index in train_indices]
	val_paths = [file_paths[index] for index in val_indices]
	val_scores = [scores[index] for index in val_indices]
	test_paths = [file_paths[index] for index in test_indices]
	test_scores = [scores[index] for index in test_indices]
	return (
		DiskDataset(train_paths, train_scores, transform=train_transform, backend='tensor'),
		DiskDataset(val_paths, val_scores, transform=eval_transform, backend='tensor'),
		DiskDataset(test_paths, test_scores, transform=eval_transform, backend='tensor'),
	)


def save_predictions(
	output_dir: Path,
	file_paths: list[Path],
	y_true: np.ndarray,
	y_pred: np.ndarray,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	with (output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as file:
		writer = csv.writer(file)
		writer.writerow(['sample_id', 'true', 'pred', 'abs_error'])
		for path, true, pred in zip(file_paths, y_true, y_pred):
			writer.writerow([path.name, f'{true:.6f}', f'{pred:.6f}', f'{abs(true - pred):.6f}'])
