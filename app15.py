import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("비디오 사물 검출 앱")

# 파일 업로드 버튼을 상단으로 이동
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# YOLO 모델 불러오기
model_path = 'hitter_trained_model.pt'  # 첨부된 모델 파일 경로
model = YOLO(model_path)

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)  # 열을 균등하게 분배하여 넓게 표시

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
        background-color: #4d4d4d;  /* 진한 회색 */
        color: #ffffff;             /* 흰색 텍스트 */
        font-weight: bold;          /* 굵은 글씨 */
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #333333;  /* 호버 시 더 진한 회색 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 사물 검출 버튼 클릭 이벤트 처리
if st.button("사물 검출 실행"):
    if uploaded_file is not None:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            output_path = temp_output.name

        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input_path = temp_input.name

        # 비디오 캡처 및 YOLO 추론
        cap = cv2.VideoCapture(temp_input_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱을 XVID로 변경
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0  # 디버깅용 프레임 카운트
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 모델로 예측 수행
            results = model(frame)
            if len(results) > 0:
                annotated_frame = results[0].plot()  # 예측 결과가 표시된 프레임
                out.write(annotated_frame)
            else:
                # 검출이 없을 경우 원본 프레임을 그대로 저장 (디버깅)
                out.write(frame)
                st.write(f"Frame {frame_count}: No detections")
            
            frame_count += 1

        cap.release()
        out.release()

        # 결과 비디오를 st.session_state에 저장하여 스트림릿에 표시
        st.session_state["processed_video"] = output_path
        result_placeholder.video(output_path)
        st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")
    else:
        st.warning("사물 검출을 실행하려면 비디오 파일을 업로드하세요.")
