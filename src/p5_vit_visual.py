import torch
import daisy
from daisy.analysis.mae import ViTAttentionRelevanceExplainer, generate_vit_attention_relevance_heatmap
from daisy.model.mae import create_vit_model
from common import LABELED_DATA_DIR, LABELED_DATA_SHEET, DATA_SPLIT_SHEET, NUM_CLASSES, OUTPUT_ROOT, load_smile
from PIL import Image
import numpy as np


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	for task in range(1, 8):
		print(f'Processing task {task}...')
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
		explainer = ViTAttentionRelevanceExplainer(model)

		test_dataset = load_smile(
			root=LABELED_DATA_DIR,
			class_sheet=LABELED_DATA_SHEET,
			split_sheet=DATA_SPLIT_SHEET,
			task=task,
		)[2]
		test_dataset.applyTransform(daisy.util.transform.get_stretch_val_transform())
		output_dir = OUTPUT_ROOT / 'p5_vit_visual' / f'task_{task}'
		output_dir.mkdir(parents=True, exist_ok=True)

		for i in range(len(test_dataset)):
			img, _ = test_dataset[i]
			with torch.no_grad():
				output = model(img.unsqueeze(0).to(device))
				pred = output.argmax(dim=1).item()
			_, overlay = generate_vit_attention_relevance_heatmap(explainer, img.to(device), target_class=pred, start_layer=0)
			overlay_img = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)
			Image.fromarray(overlay_img).save(output_dir / f'{i}_overlay.png')


if __name__ == '__main__':
	main()
