import cv2
from ultralytics import YOLO

# IP Webcam 스트리밍 URL
ip_camera_url = 'http://172.168.10.52:8001/video'

# OpenCV로 스트리밍 연결 (FFmpeg 백엔드 사용)
cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)

# YOLOv8 모델 불러오기
model = YOLO('yolov8n.pt')

# 비디오 스트림 읽기
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("스트림을 불러올 수 없습니다.")
        break

    # YOLOv8 모델 실행
    results = model(frame)

    # 결과를 시각화
    annotated_frame = results[0].plot()

    # 결과 화면 출력
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()