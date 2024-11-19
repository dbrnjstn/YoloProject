import cv2
import torch
from ultralytics import YOLO

# IP Webcam 스트리밍 URL
ip_camera_url = 'http://172.168.10.62:8001/video'

# OpenCV로 스트리밍 연결 (FFmpeg 백엔드 사용)
cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)

# YOLOv8 모델 불러오기
model = YOLO('yolov8n.pt')

# 강제로 CPU 사용 (CUDA 비활성화)
device = 'cpu'
model.to(device)

# 비디오 스트림 읽기
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("스트림을 불러올 수 없습니다.")
        break

    # 프레임 크기 조정 (BCHW 형식으로 변환)
    resized_frame = cv2.resize(frame, (640, 640))  # YOLOv8 요구 사항에 맞게 크기 조정

    # 프레임을 텐서로 변환 후 CPU로 이동 (데이터 타입을 float32로 변환)
    frame_tensor = torch.tensor(resized_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # YOLOv8 모델 실행
    results = model(frame_tensor)

    # 결과를 시각화
    annotated_frame = results[0].plot()

    # 결과 화면 출력
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
