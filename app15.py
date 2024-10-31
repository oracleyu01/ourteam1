import streamlit as st
from ultralytics import YOLO
import tempfile
import subprocess
import os
import cv2

# 페이지 설정
st.set_page_config(layout="wide")
st.title("비디오 사물 검출 앱")

# 모델 로드
model_path = st.file_uploader("모델 파일을 업로드하세요", type=["pt"])
if model_path:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        tmp_model.write(model_path.read())
        model = YOLO(tmp_model.name)
    st.success("모델이 성공적으로 로드되었습니다.")

# 비디오 업로드
uploaded_video = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 검출 결과 비디오 플레이스홀더
result_placeholder = st.empty()

# 임시 파일 경로
output_path = ""
reencoded_output_path = ""

# 사물 검출 실행 버튼
if st.button("사물 검출 실행"):
    if uploaded_video and model:
        # 비디오 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            input_video_path = tmp_video.name
        
        # 결과 비디오 임시 저장 경로
        output_path = input_video_path.replace(".mp4", "_detected.mp4")

        # OpenCV를 사용하여 비디오 처리 및 YOLO 모델 적용
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 모델 예측 수행
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    label = f"{box.label} {confidence:.2f}"
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 처리된 프레임을 결과 비디오에 작성
            out.write(frame)

        cap.release()
        out.release()
        
        # 결과 비디오를 재인코딩하여 Streamlit 호환성 확보
        reencoded_output_path = output_path.replace(".mp4", "_reencoded.mp4")
        subprocess.run(
            ["ffmpeg", "-i", output_path, "-vcodec", "libx264", "-acodec", "aac", reencoded_output_path],
            check=True
        )
        
        st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")
    else:
        st.warning("비디오 파일과 모델 파일을 모두 업로드하세요.")

# 결과 비디오 다운로드 및 재생
if os.path.exists(reencoded_output_path):
    with open(reencoded_output_path, "rb") as f:
        st.download_button(
            label="결과 비디오 다운로드",
            data=f,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )
    result_placeholder.video(reencoded_output_path)
