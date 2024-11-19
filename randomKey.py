from Crypto.Random import get_random_bytes

# 16바이트 AES 키와 IV를 랜덤하게 생성
secret_key = get_random_bytes(16)  # AES-128의 경우 16바이트, AES-256은 32바이트
iv = get_random_bytes(16)           # AES-CBC 모드에서 필요한 16바이트 IV

print("Generated Key:", secret_key.hex())
print("Generated IV:", iv.hex())
