import os

# 이미지가 저장된 디렉터리 경로
image_dir = 'dataset/images/val/'  # 적절한 경로로 수정

# YOLO 형식 라벨 파일을 저장할 디렉터리 경로
label_dir = 'dataset/labels/val/'
os.makedirs(label_dir, exist_ok=True)

# 이미지 파일에 대응하는 YOLO 라벨 파일 생성
for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        # 라벨 파일 경로
        label_file = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        
        # 라벨 파일 내용 작성
        with open(label_file, 'w') as f:
            f.write('0 0.5 0.5 1.0 1.0\n')

print("라벨 파일 생성 완료!")
