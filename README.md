# SMILE Companion

Code for MAE pretraining, seven downstream image-classification tasks, and smile-aesthetics score regression used in the SMILE Companion study. The repository uses [Daisy](https://github.com/bakabaka9405/daisy) as the shared training, model, and analysis library.

## Experimental workflow

### 1. MAE pretraining and mask-ratio selection

A ViT-Base model is pretrained with masked autoencoding. The study first compares mask ratios of 20%, 40%, 60%, and 80%, then refines the search with 65%, 70%, and 75%.

Each pretrained representation is evaluated by linear probing: the backbone is frozen and only the classification head is trained on 2,000 labeled training images. The comparison selects a mask ratio of 65% for subsequent experiments.

The repository retains the selected configuration in `src/p1_mae_pretrain.toml` and the linear-probing implementation in `src/p2_vit_linprobe_all.py`.

### 2. Data-scale studies

Using the selected 65% mask ratio, the study repeats the same pretraining and linear-probing workflow with 10%, 25%, 50%, 75%, and 100% of the unlabeled images. It also repeats full fine-tuning with 500, 1,000, 2,000, 4,000, and 8,000 labeled training images while keeping the validation set fixed.

These experiments reuse the same pretraining, probing, and fine-tuning procedures, so they are summarized rather than represented by duplicate entrypoints.

### 3. Full fine-tuning and learning-rate selection

The selected MAE checkpoint is fully fine-tuned on all seven tasks using base learning rates of `1e-4`, `2e-4`, `5e-4`, `1e-3`, `2e-3`, and `5e-3`. The effective learning rate is calculated as:

```text
lr = base_lr * batch_size / 256
```

With a batch size of 64, the study selects a base learning rate of `1e-3`. The complete sweep is implemented in `src/p3_vit_finetune_all.py`.

### 4. Adaptation-strategy comparison

The study compares full fine-tuning, LoRA fine-tuning, and linear probing. Full fine-tuning uses the selected `1e-3` setting, LoRA updates a parameter-efficient adapter, and linear probing keeps the pretrained backbone frozen.

The corresponding implementations are `src/p3_vit_finetune_all.py`, `src/p4_vit_lora_finetune.py`, and `src/p2_vit_linprobe_all.py`.

### 5. Evaluation and interpretability

The selected full fine-tuning checkpoint is evaluated on the held-out internal test split with `src/p6_vit_test.py`. Attention-relevance maps for the same ViT model are generated with `src/p5_vit_visual.py`.

The broader study also applies the selected model to external FFHQ, phone, and camera test sets. These evaluations use the same checkpoint and evaluation procedure.

### 6. Downstream regression transfer

To verify that the MAE-pretrained representation also supports a non-classification task, the selected ViT checkpoint is fine-tuned to predict the overall aesthetic score of a smile image. P7 uses a deterministic 8:1:1 training, validation, and test split. The best checkpoint is selected by validation Pearson correlation and evaluated with Pearson correlation, MAE, RMSE, and ICC(A,1) on the held-out test split.

The regression workflow is implemented in `src/p7_mae_regression.py`, with its dataset utilities isolated in `src/regression_util.py`.

## Tasks

| Task | Classes |
| --- | --- |
| 1. Smile line | high / average / low |
| 2. Smile arc | no discernible / upward-curving / flat / reverse |
| 3. Most posterior tooth displayed | none / canine / first premolar / second premolar / first molar |
| 4. Mandibular incisor display | none / partial / complete |
| 5. Lower-lip–maxillary-incisal-edge relationship | slight contact / no contact / incisal edges covered |
| 6. Upper-lip curvature | upward / straight / downward |
| 7. Buccal corridor | present / absent / unilateral |

## Environment and installation

Use Python 3.12+.

```bash
pip install -r requirements.txt
```

## Data

Arrange local data as follows:

```text
data/
├── unlabeled/
├── labeled/
│   ├── images/
│   ├── labels.xlsx
│   └── split.csv
└── regression/
    ├── images/
    └── scores.txt
```

In `split.csv`, split values are `1` for training, `2` for validation, and `3` for testing. Task columns are `1` through `7`.

Regression images are numbered consecutively, and `scores.txt` contains the corresponding aesthetic scores in the same order.

## Running experiments

Run commands from the repository root.

### MAE pretraining and linear probing

```bash
python -c "from daisy.task import run_task; run_task('src/p1_mae_pretrain.toml')"
python src/p2_vit_linprobe_all.py
```

### Fine-tuning

```bash
python src/p3_vit_finetune_all.py
python src/p4_vit_lora_finetune.py
```

### Evaluation and visualization

```bash
python src/p6_vit_test.py
python src/p5_vit_visual.py
```

### Regression transfer

```bash
python src/p7_mae_regression.py
```

Training outputs and checkpoints are written under `outputs/`.
