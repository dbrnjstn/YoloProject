import qrcode
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import base64

# JSON 데이터 생성
data = {
    "disabledPersonCarNum": "234가5678",
    "issueDate": "2024-10-11",
}

# JSON 데이터를 문자열로 변환
json_data = json.dumps(data, ensure_ascii=False)

# 16바이트 암호화 키와 IV 설정
secret_key = b'452fc981217ab2b2'  # 16바이트 키
iv = b'956914c27b3f7490'           # 16바이트 IV

# AES 암호화 (CBC 모드 사용)
cipher = AES.new(secret_key, AES.MODE_CBC, iv)
encrypted_data = cipher.encrypt(pad(json_data.encode('utf-8'), AES.block_size))

# 암호화된 데이터를 URL-safe Base64로 인코딩하여 QR 코드에 삽입
encrypted_data_b64 = base64.urlsafe_b64encode(encrypted_data).decode('utf-8')

# QR 코드 생성
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(encrypted_data_b64)
qr.make(fit=True)

# QR 코드 이미지 생성 및 저장
img = qr.make_image(fill="black", back_color="white")
img.save("secretqrcode3.png")
