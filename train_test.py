from ultralytics import YOLO

# 학습된 모델 불러오기
model = YOLO('runs/detect/train12/weights/best.pt')

# 테스트 이미지 경로
test_image_path = 'images/car.jpg'

# 예측 수행
results = model(test_image_path)

# 결과 시각화
results[0].show()  # 리스트의 첫 번째 결과에 대해 show() 호출
