import torch
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# YOLOv8 모델 로드 (\"yolov8n.pt\"은 pretrained된 모델 파일 경로입니다)
model = YOLO("yolov8n.pt")

# IP Webcam 스트리밍 URL
ip_camera_url = 'http://172.168.10.52:8001/video'

# OpenCV로 스트리밍 연결 (FFmpeg 백엔드 사용)
cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open IP Webcam stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("스트림을 불러올 수 없습니다.")
        break

    # YOLOv8을 이용해 객체 탐지 수행
    results = model(frame)

    # 탐지된 객체 중에서 차량으로 판단되는 객체 추출
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls.item()
            if cls in [2, 5, 7]:  # 2: 자동차, 5: 버스, 7: 트럭 (예시)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_img = frame[y1:y2, x1:x2]

                # 차량 이미지에서 번호판을 찾기 위한 이미지 전처리
                gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blurred, 100, 200)

                # 윤곽선을 찾아 번호판 후보 영역을 탐색
                contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if 2 < aspect_ratio < 5:  # 번호판의 일반적인 가로세로 비율
                        plate_img = vehicle_img[y:y+h, x:x+w]

                        # pytesseract를 사용한 번호판 텍스트 인식
                        plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                        _, plate_img_thresh = cv2.threshold(plate_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        plate_text = pytesseract.image_to_string(plate_img_thresh, config='--psm 7')
                        print("Detected License Plate: {plate_text}")

                        # 탐지된 번호판 영역 표시
                        cv2.rectangle(frame, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x1 + x, y1 + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 영상 출력
    cv2.imshow("YOLOv8 License Plate Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()