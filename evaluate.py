from ultralytics import YOLO

MODEL_PATH = "runs/detect/wood_defect_yolov8s/weights/best.pt"

model = YOLO(MODEL_PATH)
if __name__ == '__main__':
	metrics = model.val(
			data="data.yaml",
			split="test",
			imgsz=640,
			conf=0.25,
			name="wood_defect_test_evaluation"
	)

	print("Precision:", metrics.box.mp)
	print("Recall:", metrics.box.mr)
	print("mAP50:", metrics.box.map50)
	print("mAP50-95:", metrics.box.map)