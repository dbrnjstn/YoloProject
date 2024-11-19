from ultralytics import YOLO

if __name__ == '__main__':
    # YOLO 모델 불러오기
    model = YOLO('yolov8n.yaml')  # 처음부터 학습하려면 YAML 파일 사용, 프리트레인된 가중치를 사용하려면 'yolov8n.pt' 사용

    # 데이터셋 YAML 파일 경로 설정
    data_path = 'dataset.yaml'  # dataset.yaml 파일 경로

    # 모델 학습 시작                                                        
    model.train(data=data_path, epochs=100, batch=8, imgsz=416, save_dir='D:/yolo_training_results', amp=False, resume=True)
