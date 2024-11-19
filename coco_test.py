import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(layout='wide')

def main():
    st.title("자동차 번호판 인식")

    # Streamlit에서 URL 파라미터 가져오기
    query_params = st.experimental_get_query_params()
    image_url = query_params.get("image_url", [None])[0]

    if image_url:
        st.write("이미지 URL을 통해 이미지를 받았습니다.")
        # 이미지 유효성 확인
        is_valid, image = validate_image_url(image_url)
        if is_valid:
            st.image(image, caption="업로드된 이미지", use_column_width=True)
            # 여기서 YOLO 모델 등을 사용하여 추가 처리를 진행할 수 있습니다.
        else:
            st.write("유효하지 않은 이미지 URL입니다.")
    else:
        st.write("이미지가 업로드되지 않았습니다.")

def validate_image_url(url):
    try:
        # HTTP GET 요청을 통해 이미지 URL 확인 (SSL 인증서 검증 비활성화)
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            # 응답 데이터를 이미지로 열어보기
            image = Image.open(BytesIO(response.content))
            return True, image
        else:
            return False, None
    except Exception as e:
        # 요청 실패 시 오류 메시지 출력
        st.write(f"이미지 URL 확인 중 오류 발생: {e}")
        return False, None

if __name__ == '__main__':
    main()
