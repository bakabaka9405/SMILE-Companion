from pathlib import Path
from typing import Literal

import daisy
from daisy.dataset import DiskDataset, MemoryDataset
from daisy.util import extract_by_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELED_DATA_DIR = PROJECT_ROOT / 'data' / 'labeled' / 'images'
LABELED_DATA_SHEET = PROJECT_ROOT / 'data' / 'labeled' / 'labels.xlsx'
DATA_SPLIT_SHEET = PROJECT_ROOT / 'data' / 'labeled' / 'split.csv'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
NUM_CLASSES = [0, 3, 4, 5, 3, 3, 3, 3]


def load_smile(
	root: Path,
	class_sheet: Path,
	split_sheet: Path,
	task: int,
	backend: Literal['pil', 'tensor'] = 'tensor',
) -> tuple[DiskDataset, DiskDataset, MemoryDataset]:
	feeder = daisy.feeder.load_feeder_from_sheet(
		dataset_root=root,
		sheet_path=class_sheet,
		column=task,
		have_header=True,
		label_offset=(0 if task in [2, 3] else -1),
	)

	files, labels = feeder.fetch()
	split_feeder = daisy.feeder.load_feeder_from_sheet(
		dataset_root=root,
		sheet_path=split_sheet,
		column=task,
		have_header=True,
	)
	split_files, split_labels = split_feeder.fetch()
	for data_file, split_file in zip(files, split_files):
		assert data_file.stem == split_file.stem, f'{data_file.stem} != {split_file.stem}'
	train_files, train_labels = extract_by_label(split_labels, 1, files, labels)
	train_dataset = DiskDataset(train_files, train_labels, backend=backend)
	val_files, val_labels = extract_by_label(split_labels, 2, files, labels)
	val_dataset = DiskDataset(val_files, val_labels, backend=backend)
	test_files, test_labels = extract_by_label(split_labels, 3, files, labels)
	test_dataset = MemoryDataset(test_files, test_labels, backend=backend)
	return train_dataset, val_dataset, test_dataset
