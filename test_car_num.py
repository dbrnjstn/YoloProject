import cv2
from ultralytics import YOLO
import pytesseract

# Tesseract 경로 설정 (Windows에서 필요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR로 번호판 텍스트 인식 함수
def recognize_license_plate(license_plate_img):
    # 이미지를 흑백으로 변환
    gray_img = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

    # OCR로 텍스트 인식
    text = pytesseract.image_to_string(gray_img, config='--psm 7')  # psm 7: 단일 텍스트 줄 모드
    return text

# YOLOv8 모델 불러오기
model = YOLO('yolov8n.pt')  # yolov8n.pt 대신 사용할 모델 경로를 지정하세요

# 차량 이미지 읽기
image_path = r'C:\Users\LX\yolo_training\images/car.jpg'  # 이미지 경로를 수정하세요
img = cv2.imread(image_path)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
else:
    # YOLOv8을 사용해 번호판 탐지
    results = model(img)

    # 탐지된 객체 중 'license_plate'를 찾기
    for result in results:
        for box in result.boxes:
            class_id = box.cls.cpu().numpy()[0]
            class_name = model.names[class_id]  # 클래스 이름
            if class_name == 'license_plate':  # 번호판 클래스 탐지
                print(f"번호판 탐지됨: {box.xywh[0].cpu().numpy()}")

                # 번호판 영역 자르기
                x, y, w, h = box.xywh[0].cpu().numpy()
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)  # 좌표 변환
                license_plate_img = img[y:y+h, x:x+w]

                # 번호판 텍스트 인식
                license_plate_text = recognize_license_plate(license_plate_img)
                print(f"인식된 번호판: {license_plate_text}")

                # 번호판 이미지를 출력하여 확인
                cv2.imshow('Detected License Plate', license_plate_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break
