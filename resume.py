from ultralytics import YOLO
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

model = YOLO("runs/detect/wood_defect_yolov8n_grouped-2/weights/last.pt")
if __name__ == '__main__':

	model.train(
			resume=True
	)