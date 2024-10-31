import cv2
import streamlit as st
from streamlit import runtime

st.title("Real-Time Webcam Stream")

# 웹캠 열기
camera = cv2.VideoCapture(0)

# 스트리밍 상태 확인
if runtime.exists():
    while True:
        ret, frame = camera.read()  # 프레임 읽기
        if not ret:
            st.write("카메라에서 영상을 가져올 수 없습니다.")
            break

        # BGR에서 RGB로 변환 (streamlit 이미지 렌더링을 위해)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 이미지를 Streamlit에 표시
        st.image(frame, channels="RGB")

# 웹캠 릴리스
camera.release()
