import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from imutils.object_detection import non_max_suppression
from ultralytics import YOLO

min_confidence = 0.5
file_name = "images/car2.jpg"
frame_size = 320
padding = 0.05

# Load YOLOv8 모델 (사전 학습된 모델 사용)
model = YOLO('yolov8n.pt')

def carROI(image):
    height, width, channels = image.shape

    # YOLOv8으로 객체 탐지 실행
    results = model(image)

    # 탐지된 객체 정보 저장
    boxes = []
    confidences = []
    img_cars = []

    for result in results:
        for box in result.boxes:
            class_id = box.cls.cpu().numpy()[0]  # 클래스 ID
            confidence = box.conf.cpu().numpy()[0]  # 신뢰도
            x, y, w, h = box.xywh[0].cpu().numpy()  # 좌표 정보

            if class_id == 2 and confidence > min_confidence:  # 'car' 클래스 확인
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                img_cars.append(image[y:y+h, x:x+w])

                # 차량 탐지에 성공한 경우 반환
                return (boxes[0], image[y:y+h, x:x+w])

    # 차량이 탐지되지 않았을 경우 None 반환
    return None, None

def textROI(image):
    # 이미지 크기 재조정
    orig = image.copy()
    (origH, origW) = image.shape[:2]
    rW = origW / float(frame_size)
    rH = origH / float(frame_size)
    newW = int(origW / rH)
    center = int(newW / 2)
    start = center - int(frame_size / 2)

    image = cv2.resize(image, (newW, frame_size))
    scale_image = image[0:frame_size, start:start+frame_size]
    (H, W) = scale_image.shape[:2]

    # EAST Text Detector 모델 불러오기
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # Blob 생성 및 전처리
    blob = cv2.dnn.blobFromImage(image, 1.0, (frame_size, frame_size), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # 텍스트 영역 감지
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # 최종 텍스트 영역 반환
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        return ([startX, startY, endX, endY], orig[startY:endY, startX:endX])

def textRead(image):
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(image, config=config)
    text = "".join([c if c.isalnum() else "" for c in text]).strip()
    print(f"OCR TEXT: {text}")
    return text

# 메인 실행 부분
img = cv2.imread(file_name)
img_copy = img.copy()

(car_box, car_image) = carROI(img)

if car_box is None:
    print("차량을 탐지하지 못했습니다.")
else:
    (x, y, w, h) = car_box
    ([startX, startY, endX, endY], text_image) = textROI(car_image)

    # 텍스트 영역에서 OCR 실행
    text = textRead(text_image)

    # 텍스트 위치 및 결과 표시
    cv2.rectangle(img_copy, (x+startX, y+startY), (x+endX, y+endY), (0, 255, 0), 2)
    cv2.putText(img_copy, text, (x+startX, y+startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 결과 이미지 시각화
    cv2.imshow("OCR Text Recognition: " + text, img_copy)
    cv2.imshow('Plate Image', text_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
