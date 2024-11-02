import streamlit as st
import subprocess
import sys
import tempfile
import os
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# 시스템 패키지 설치 시도 (libGL 관련 오류 해결)
try:
    import cv2
except ImportError:
    st.write("필수 시스템 라이브러리를 설치하는 중입니다...")
    # 시스템 패키지 업데이트 및 libGL 설치
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "libgl1-mesa-glx"])
    # OpenCV 헤드리스 버전 설치
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("비디오 사물 검출 앱")

# 모델 파일 업로드
model_file = st.file_uploader("모델 파일을 업로드하세요", type=["pt"])
if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(model_file.read())
        model_path = temp_model_file.name
    model = YOLO(model_path)
    st.success("모델이 성공적으로 로드되었습니다.")

# 비디오 파일 업로드
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("사물 검출 결과 영상")
        result_placeholder = st.empty()
        if "processed_video" in st.session_state and st.session_state["processed_video"] is not None:
            result_placeholder.video(st.session_state["processed_video"])
        else:
            result_placeholder.markdown(
                """
                <div style='width:100%; height:620px; background-color:#d3d3d3; display:flex; align-items:center; justify-content:center; border-radius:5px;'>
                    <p style='color:#888;'>여기에 사물 검출 결과가 표시됩니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 버튼 스타일 설정
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4d4d4d;
        color: #ffffff;
        font-weight: bold;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 사물 검출 버튼 클릭 이벤트 처리
if st.button("사물 검출 실행") and uploaded_file and model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    cap = cv2.VideoCapture(temp_input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 예측 수행 및 디버깅
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        if len(detections) > 0:
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            st.write(f"프레임 {frame_count}: {len(detections)}개 검출됨")
        else:
            st.write(f"프레임 {frame_count}: 검출 없음 - 원본 프레임 저장")

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # moviepy로 재인코딩하여 저장
    reencoded_path = output_path.replace(".mp4", "_reencoded.mp4")
    clip = VideoFileClip(output_path)
    clip.write_videofile(reencoded_path, codec="libx264", audio_codec="aac")

    # 결과 비디오를 st.session_state에 저장하여 스트림릿에 표시
    st.session_state["processed_video"] = reencoded_path
    result_placeholder.video(reencoded_path)
    st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")

    # 다운로드 링크 제공
    with open(reencoded_path, "rb") as file:
        st.download_button(
            label="결과 영상 다운로드",
            data=file,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )